import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import gc
from torch.utils.checkpoint import checkpoint
from datetime import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix

# ============================================================================
# ARCHITECTURE: ScalogramV4_STGAT (With Memory Workarounds)
# ============================================================================

class PhysicsDeterministicGate(nn.Module):
    def __init__(self, kp_threshold=5.0):
        super(PhysicsDeterministicGate, self).__init__()
        self.kp_threshold = kp_threshold
    def forward(self, space_weather):
        kp_index = space_weather[:, 0]
        gate_val = torch.sigmoid(-2.0 * (kp_index - self.kp_threshold))
        return gate_val.unsqueeze(1)

class SpatialGNNModule(nn.Module):
    def __init__(self, in_features=512, hidden=256, out_features=512):
        super().__init__()
        self.gat1 = nn.Linear(in_features, hidden)
        self.gat2 = nn.Linear(hidden, out_features)
        self.attention = nn.Parameter(torch.ones(24))
    def forward(self, x):
        h = F.relu(self.gat1(x))
        h = self.gat2(h)
        weights = F.softmax(self.attention, dim=0)
        consensus = torch.sum(h * weights.view(1, 24, 1), dim=1)
        return consensus, weights

class ScalogramV4_STGAT(nn.Module):
    def __init__(self, n_regions=6, deterministic_gate=False):
        super().__init__()
        efficientnet = models.efficientnet_b1(weights=None)
        self.spatial_encoder = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.temporal_encoder = nn.GRU(
            input_size=1280, hidden_size=256, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        self.gat_layer = SpatialGNNModule(in_features=512, hidden=256, out_features=512)
        
        if deterministic_gate:
            self.cosmic_gate = PhysicsDeterministicGate(kp_threshold=5.0)
        else:
            self.cosmic_gate = nn.Sequential(
                nn.Linear(2, 32), nn.ReLU(),
                nn.Linear(32, 512), nn.Sigmoid()
            )
            
        self.alpha_regions = nn.Parameter(torch.randn(n_regions))
        self.head_detect = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.head_mag = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.head_azm = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x_img, x_cosmic):
        B, N, C, H, W = x_img.shape
        x = x_img.view(B * N, C, H, W)
        
        # CHUNKED SPATIAL ENCODING (Memory Workaround)
        if self.training:
            chunk_size = 4
            feats = []
            for i in range(0, B * N, chunk_size):
                x_chunk = x[i:i+chunk_size]
                f_chunk = checkpoint(self.spatial_encoder, x_chunk, use_reentrant=False)
                feats.append(f_chunk)
            feat = torch.cat(feats, dim=0)
        else:
            feat = self.spatial_encoder(x)
            
        feat = self.pool(feat).squeeze(2).permute(0, 2, 1)
        self.temporal_encoder.flatten_parameters()
        gru_out, _ = self.temporal_encoder(feat)
        v_img = torch.mean(gru_out, dim=1)
        v_img = v_img.view(B, N, 512)
        v_consensus, att_weights = self.gat_layer(v_img)
        
        raw_pred_event = self.head_detect(v_consensus).squeeze(-1)
        gate_val = self.cosmic_gate(x_cosmic)
        v_fusion = v_consensus * gate_val
        out_mag = self.head_mag(v_fusion).squeeze(-1)
        out_azm = self.head_azm(v_fusion)
        return raw_pred_event, out_mag, out_azm, self.alpha_regions, gate_val

# ============================================================================
# DATASET & LOSS
# ============================================================================

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda1=2.0, lambda2=0.1, lambda3=0.05, lambda4=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, preds, targets):
        raw_pred_event, pred_mag, pred_azm, pred_alpha, gate_val = preds
        y_event, y_mag, y_azm, y_dist, region_idx = targets
        loss_bce = self.bce_loss(raw_pred_event, y_event)
        mask = (y_event == 1.0).float()
        num_events = torch.sum(mask) + 1e-8
        loss_mag = torch.sum(self.mse_loss(pred_mag, y_mag) * mask) / num_events
        target_rad = y_azm * (np.pi / 180.0)
        y_azm_vec = torch.stack([torch.sin(target_rad), torch.cos(target_rad)], dim=1)
        loss_azm = torch.sum(torch.mean(self.mse_loss(pred_azm, y_azm_vec), dim=1) * mask) / num_events
        alpha_batch = F.softplus(pred_alpha[region_idx])
        attenuated_energy_pred = pred_mag.unsqueeze(1) * torch.exp(-alpha_batch.unsqueeze(1) * y_dist)
        attenuated_energy_true = y_mag.unsqueeze(1) * torch.exp(-alpha_batch.unsqueeze(1) * y_dist)
        loss_phys = torch.sum(torch.mean(self.mse_loss(attenuated_energy_pred, attenuated_energy_true), dim=1) * mask) / num_events
        total_loss = (self.lambda1 * loss_bce) + (self.lambda2 * loss_mag) + \
                     (self.lambda3 * loss_azm) + (self.lambda4 * loss_phys)
        return total_loss

STATION_REGIONS = {
    'SBG': 0, 'SCN': 0, 'KPY': 0, 'GSI': 0, 'MLB': 0,
    'LWA': 1, 'LPS': 1, 'SRG': 1, 'SKB': 1, 'CLP': 1, 'YOG': 1, 'TRT': 1,
    'LUT': 2, 'ALR': 2,
    'TNT': 3, 'TND': 3, 'GTO': 3, 'LWK': 3, 'PLU': 3, 'AMB': 3, 'SMI': 3,
    'SRO': 4, 'JYP': 4,
    'TRD': 5
}

class SpatioTemporalDatasetAdapt(Dataset):
    def __init__(self, h5_path, split='val', indices=None):
        self.h5_path = h5_path
        self.split = split
        self.indices = indices
        with h5py.File(h5_path, 'r') as f:
            self.stations = [s.decode() if isinstance(s, bytes) else s for s in f[split].attrs['stations']]
            self.n_samples = len(indices) if indices is not None else len(f[split]['dates'])
            self.dates = [d.decode() if isinstance(d, bytes) else d for d in f[split]['dates']]
            
        self.station_to_region = [STATION_REGIONS.get(s, 0) for s in self.stations]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        with h5py.File(self.h5_path, 'r') as f:
            g = f[self.split]
            tensor = torch.from_numpy(g['tensors'][real_idx]).float()
            cosmic = torch.from_numpy(g['cosmic_features'][real_idx]).float()
            y_event = torch.tensor(g['label_event'][real_idx]).float()
            y_mag = torch.tensor(g['label_mag'][real_idx]).float()
            azms = g['label_azm'][real_idx]
            y_azm = torch.tensor(np.median(azms[azms > 0])).float() if np.any(azms > 0) else torch.tensor(0.0).float()
            y_dist = torch.from_numpy(g['label_dist'][real_idx]).float()
            if y_event == 1.0:
                nearest_idx = np.argmin(g['label_dist'][real_idx])
                region_idx = self.station_to_region[nearest_idx]
            else:
                region_idx = 0
            date = self.dates[real_idx]
        return tensor, cosmic, y_event, y_mag, y_azm, y_dist, region_idx, date

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_adaptation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Domain Adaptation starting on {device}...")

    # 1. Data Split (2024-2025 Val Set)
    h5_path = 'dataset_v13_train_val_M5_patched.h5'
    with h5py.File(h5_path, 'r') as f:
        dates = [d.decode() if isinstance(d, bytes) else d for d in f['val']['dates']]
    
    idx_train = [i for i, d in enumerate(dates) if '2024-01-01' <= d <= '2024-12-31']
    idx_val = [i for i, d in enumerate(dates) if '2025-01-01' <= d <= '2025-03-31']
    
    train_dataset = SpatioTemporalDatasetAdapt(h5_path, 'val', indices=idx_train)
    val_dataset = SpatioTemporalDatasetAdapt(h5_path, 'val', indices=idx_val)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print(f"Adapt-Train: {len(idx_train)} days. Adapt-Val: {len(idx_val)} days.")

    # 2. Model & Optimizer
    model = ScalogramV4_STGAT(n_regions=6).to(device)
    CKPT_V2 = 'checkpoints/best_stgat_v2.pth'
    if os.path.exists(CKPT_V2):
        print(f"Resuming from {CKPT_V2}...")
        model.load_state_dict(torch.load(CKPT_V2, map_location=device), strict=False)
    
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    criterion = PhysicsInformedLoss().to(device)

    # 3. Training Loop (Continuous Learning)
    best_val_loss = float('inf')
    patience = 4
    bad_epochs = 0
    best_epoch = 0

    print("\nPhase 1: Continuous Learning Adaptation...")
    try:
        for epoch in range(15):
            model.train()
            # BatchNorm in eval mode for stability with BS=1
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()
            
            epoch_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x_img, x_cosm, y_ev, y_mg, y_az, y_ds, r_id, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                optimizer.zero_grad()
                preds = model(x_img, x_cosm)
                loss = criterion(preds, (y_ev, y_mg, y_az, y_ds, r_id))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                gc.collect()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_img, x_cosm, y_ev, y_mg, y_az, y_ds, r_id, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                    preds = model(x_img, x_cosm)
                    v_loss = criterion(preds, (y_ev, y_mg, y_az, y_ds, r_id))
                    val_loss += v_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), 'checkpoints/stgat_domain_adapted.pth')
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print("Early Stopping triggered.")
                    break
    except Exception as e:
        print(f"Training interrupted: {e}")

    # 4. Automated Inference on Blind Test 2026
    print("\nPhase 2: Final Evaluation on Blind Test 2026...")
    # Load adapted weights
    model.load_state_dict(torch.load('checkpoints/stgat_domain_adapted.pth', map_location=device))
    # Inject Deterministic Gate
    model.cosmic_gate = PhysicsDeterministicGate(kp_threshold=5.0).to(device)
    model.eval()

    test_h5 = 'dataset_v13_blindtest_M5_patched.h5'
    test_dataset = SpatioTemporalDatasetAdapt(test_h5, 'test') # Simple wrapper
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    y_true_all = []
    y_prob_all = []
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="2026 Evaluation"):
            x_img, x_cosm, y_ev, y_mg, y_az, y_ds, r_id, date = batch
            x_img, x_cosm = x_img.to(device), x_cosm.to(device)
            
            raw_logits, _, _, _, gate_val = model(x_img, x_cosm)
            prob_final = torch.sigmoid(raw_logits).item() * gate_val.item()
            
            y_true_all.append(y_ev.item())
            y_prob_all.append(prob_final)
            results.append({
                'Date': date[0],
                'Actual': int(y_ev.item()),
                'Prob': prob_final
            })

    # Metrics
    auc = roc_auc_score(y_true_all, y_prob_all)
    tn, fp, fn, tp = confusion_matrix(y_true_all, (np.array(y_prob_all) >= 0.5).astype(int)).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("\n" + "="*50)
    print("=== CONTINUOUS LEARNING & DOMAIN ADAPTATION RESULTS ===")
    print(f"1. Adaptation Phase (2024-2025 Data)")
    print(f"   - Best Validation Epoch : {best_epoch}")
    print(f"   - Final Validation Loss : {best_val_loss:.4f}")
    print(f"\n2. 2026 Blind Test (Target: AUC > 0.70)")
    print(f"   - ROC-AUC Score         : {auc:.4f}")
    print(f"   - TPR (Sensitivity)     : {tpr:.2%}")
    print(f"   - FPR (False Alarm)     : {fpr:.2%}")
    print("="*50)

    # Save CSV
    pd.DataFrame(results).to_csv('doc/2026_blindtest_adapted.csv', index=False)
    print("Predictions saved to doc/2026_blindtest_adapted.csv")

if __name__ == '__main__':
    run_adaptation()
