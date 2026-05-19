import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
from datetime import datetime
import gc

# ============================================================================
# ARCHITECTURE: ScalogramV4_STGAT (V2: Decoupled Gating)
# ============================================================================

class SpatialGNNModule(nn.Module):
    """Simplified GAT-based fusion for multi-station scalograms."""
    def __init__(self, in_features=512, hidden=256, out_features=512):
        super().__init__()
        self.gat1 = nn.Linear(in_features, hidden)
        self.gat2 = nn.Linear(hidden, out_features)
        self.attention = nn.Parameter(torch.ones(24)) # Simple station attention

    def forward(self, x):
        # x: (B, 24, in_features)
        h = F.relu(self.gat1(x))
        h = self.gat2(h)
        # Global pooling with attention weights
        weights = F.softmax(self.attention, dim=0)
        consensus = torch.sum(h * weights.view(1, 24, 1), dim=1)
        return consensus, weights

class ScalogramV4_STGAT(nn.Module):
    def __init__(self, n_regions=6):
        super().__init__()
        # 1. Spatial Encoder (EfficientNet-B1)
        efficientnet = models.efficientnet_b1(weights=None)
        self.spatial_encoder = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        
        # 2. Temporal Encoder (BiGRU)
        self.temporal_encoder = nn.GRU(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 3. GNN Layer
        self.gat_layer = SpatialGNNModule(in_features=512, hidden=256, out_features=512)
        
        # 4. Cosmic Gate
        self.cosmic_gate = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 512), nn.Sigmoid()
        )
        
        # 5. Physics Parameters
        self.alpha_regions = nn.Parameter(torch.randn(n_regions))
        
        # 6. Task Heads
        self.head_detect = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1) # Raw logits
        )
        self.head_mag = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.head_azm = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 2) # [sin, cos]
        )

    def forward(self, x_img, x_cosmic):
        # x_img: (B, 24, 3, 128, 1440)
        B, N, C, H, W = x_img.shape
        x = x_img.view(B * N, C, H, W)
        
        # Spatial Encoding
        feat = self.spatial_encoder(x)
        feat = self.pool(feat).squeeze(2).permute(0, 2, 1) # (B*N, Seq, 1280)
        
        # Temporal Encoding
        self.temporal_encoder.flatten_parameters()
        gru_out, _ = self.temporal_encoder(feat)
        v_img = torch.mean(gru_out, dim=1) # (B*N, 512)
        
        v_img = v_img.view(B, N, 512)
        
        # Graph Attention Fusion
        v_consensus, att_weights = self.gat_layer(v_img)
        
        # V2: Bypass Detection Head (Calculated from ungated features)
        raw_pred_event = self.head_detect(v_consensus).squeeze(-1)
        
        # Cosmic Gating (Affects regression tasks)
        gate_val = self.cosmic_gate(x_cosmic)
        v_fusion = v_consensus * gate_val
        
        # Heads
        out_mag = self.head_mag(v_fusion).squeeze(-1)
        out_azm = self.head_azm(v_fusion)
        
        return raw_pred_event, out_mag, out_azm, self.alpha_regions, gate_val

# ============================================================================
# LOSS FUNCTION: PhysicsInformedLoss (V2: Priority & Gating Bypass)
# ============================================================================

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda1=2.0, lambda2=0.1, lambda3=0.05, lambda4=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.lambda1 = lambda1 # Higher priority for Detection
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, preds, targets):
        # V2: preds has 5 elements
        raw_pred_event, pred_mag, pred_azm, pred_alpha, gate_val = preds
        y_event, y_mag, y_azm, y_dist, region_idx = targets

        # 1. BCE Loss (Using RAW logits to bypass gating gradients)
        loss_bce = self.bce_loss(raw_pred_event, y_event)

        # MASKING: Regresi & Fisika hanya dihitung jika ada Event Asli (Mw >= 5.0)
        mask = (y_event == 1.0).float()
        num_events = torch.sum(mask) + 1e-8

        # 2. MSE Loss (Magnitude & Azimuth)
        loss_mag = torch.sum(self.mse_loss(pred_mag, y_mag) * mask) / num_events
        
        target_rad = y_azm * (np.pi / 180.0)
        y_azm_vec = torch.stack([torch.sin(target_rad), torch.cos(target_rad)], dim=1)
        loss_azm = torch.sum(torch.mean(self.mse_loss(pred_azm, y_azm_vec), dim=1) * mask) / num_events

        # 3. Physics Loss (DPINN Attenuation Law)
        alpha_batch = F.softplus(pred_alpha[region_idx])
        attenuated_energy_pred = pred_mag.unsqueeze(1) * torch.exp(-alpha_batch.unsqueeze(1) * y_dist)
        attenuated_energy_true = y_mag.unsqueeze(1) * torch.exp(-alpha_batch.unsqueeze(1) * y_dist)
        
        loss_phys = torch.sum(torch.mean(self.mse_loss(attenuated_energy_pred, attenuated_energy_true), dim=1) * mask) / num_events

        # 4. Total Loss
        total_loss = (self.lambda1 * loss_bce) + (self.lambda2 * loss_mag) + \
                     (self.lambda3 * loss_azm) + (self.lambda4 * loss_phys)

        return total_loss, {'bce': loss_bce.item(), 'mag': loss_mag.item(), 'azm': loss_azm.item(), 'phys': loss_phys.item()}

# ============================================================================
# DATASET & UTILS
# ============================================================================

STATION_REGIONS = {
    'SBG': 0, 'SCN': 0, 'KPY': 0, 'GSI': 0, 'MLB': 0,
    'LWA': 1, 'LPS': 1, 'SRG': 1, 'SKB': 1, 'CLP': 1, 'YOG': 1, 'TRT': 1,
    'LUT': 2, 'ALR': 2,
    'TNT': 3, 'TND': 3, 'GTO': 3, 'LWK': 3, 'PLU': 3, 'AMB': 3, 'SMI': 3,
    'SRO': 4, 'JYP': 4,
    'TRD': 5
}

class SpatioTemporalDataset(Dataset):
    def __init__(self, h5_path, split='train'):
        self.h5_path = h5_path
        self.split = split
        with h5py.File(h5_path, 'r') as f:
            self.n_samples = len(f[split]['dates'])
            self.stations = [s.decode() if isinstance(s, bytes) else s for s in f[split].attrs['stations']]
            
        self.station_to_region = [STATION_REGIONS.get(s, 0) for s in self.stations]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            g = f[self.split]
            tensor = torch.from_numpy(g['tensors'][idx]).float()
            cosmic = torch.from_numpy(g['cosmic_features'][idx]).float()
            y_event = torch.tensor(g['label_event'][idx]).float()
            y_mag = torch.tensor(g['label_mag'][idx]).float()
            y_azm = torch.tensor(0.0).float()
            azms = g['label_azm'][idx]
            if y_event == 1.0:
                y_azm = torch.tensor(np.median(azms[azms > 0])).float() if np.any(azms > 0) else torch.tensor(0.0).float()
            y_dist = torch.from_numpy(g['label_dist'][idx]).float()
            if y_event == 1.0:
                nearest_idx = np.argmin(g['label_dist'][idx])
                region_idx = self.station_to_region[nearest_idx]
            else:
                region_idx = 0
        return tensor, cosmic, y_event, y_mag, y_azm, y_dist, region_idx

# ============================================================================
# MAIN TRAINING PIPELINE V2
# ============================================================================

def train_stgat_v2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ensure directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # 1. Initialize Model
    model = ScalogramV4_STGAT(n_regions=6).to(device)
    
    # TAHAP 1: MODEL SURGERY
    V3_CKPT = 'd:/multi/scalogramv3/checkpoints/v3_v8_best.pth'
    if os.path.exists(V3_CKPT):
        print(f"Loading V3 weights from {V3_CKPT}...")
        v3_state = torch.load(V3_CKPT, map_location='cpu')['model_state_dict']
        new_state = model.state_dict()
        mapping = {'features': 'spatial_encoder', 'gru': 'temporal_encoder', 'head_detection': 'head_detect', 'head_magnitude': 'head_mag'}
        matches = 0
        for v3_key, v4_key_prefix in mapping.items():
            for k in v3_state.keys():
                if k.startswith(v3_key):
                    new_key = k.replace(v3_key, v4_key_prefix)
                    if new_key in new_state:
                        if v3_state[k].shape == new_state[new_key].shape:
                            new_state[new_key] = v3_state[k]
                            matches += 1
        model.load_state_dict(new_state, strict=False)
        print(f"Successfully matched and transferred {matches} parameter tensors.")
    else:
        print("V3 Checkpoint not found. Starting from scratch.")

    # Data
    dataset_path = 'dataset_v13_train_val_M5_patched.h5'
    train_loader = DataLoader(SpatioTemporalDataset(dataset_path, 'train'), batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(SpatioTemporalDataset(dataset_path, 'val'), batch_size=1, num_workers=0, pin_memory=False)

    criterion = PhysicsInformedLoss().to(device)
    
    # TAHAP 3: FASE 1 TRAINING (FREEZE & WARM-UP)
    print("\nFase 1 (Warm-up) Dimulai...")
    for param in model.spatial_encoder.parameters(): param.requires_grad = False
    for param in model.temporal_encoder.parameters(): param.requires_grad = False
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    history = {'total': [], 'bce': [], 'mag': [], 'azm': [], 'phys': []}
    
    try:
        for epoch in range(15): # 15 Epochs Warm-up
            model.train()
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()
            
            epoch_loss = 0
            for i, batch in enumerate(tqdm(train_loader, desc=f"Phase 1 Epoch {epoch+1}")):
                x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, r_idx = [b.to(device) for b in batch]
                optimizer.zero_grad()
                preds = model(x_img, x_cosmic)
                loss, logs = criterion(preds, (y_event, y_mag, y_azm, y_dist, r_idx))
                
                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                
                if (i + 1) % 10 == 0:
                    print(f"  Iter {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
                del loss, preds, x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, r_idx
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            print(f"Phase 1 Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
            gc.collect()

        torch.save(model.state_dict(), 'checkpoints/stgat_v2_phase1_complete.pth')

        # TAHAP 4: FASE 2 TRAINING (GLOBAL FINE-TUNING)
        print("\nFase 2 (Global Fine-tuning) Dimulai...")
        # V2: Unfreeze ALL parameters as requested
        for param in model.parameters(): param.requires_grad = True
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        best_val_loss = float('inf')
        for epoch in range(30): # 30 Epochs Fine-tuning
            model.train()
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()
            
            epoch_loss = 0
            for i, batch in enumerate(tqdm(train_loader, desc=f"Phase 2 Epoch {epoch+1}")):
                x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, r_idx = [b.to(device) for b in batch]
                optimizer.zero_grad()
                preds = model(x_img, x_cosmic)
                loss, logs = criterion(preds, (y_event, y_mag, y_azm, y_dist, r_idx))
                
                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                
                history['total'].append(loss.item())
                history['bce'].append(logs['bce'])
                history['mag'].append(logs['mag'])
                history['azm'].append(logs['azm'])
                history['phys'].append(logs['phys'])
                
                del loss, preds, x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, r_idx
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
            scheduler.step()
            print(f"Phase 2 Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
            torch.save(model.state_dict(), f'checkpoints/stgat_v2_last.pth')
            if (epoch_loss/len(train_loader)) < best_val_loss:
                best_val_loss = epoch_loss/len(train_loader)
                torch.save(model.state_dict(), f'checkpoints/best_stgat_v2.pth')
            gc.collect()

        torch.save(model.state_dict(), 'checkpoints/best_stgat_v2.pth')

    except Exception as e:
        with open('logs/stgat_v2_crash_report.txt', 'a') as f:
            f.write(f"\n[{datetime.now()}] CRASH DETECTED\n")
            f.write(traceback.format_exc())
        torch.save(model.state_dict(), 'checkpoints/stgat_v2_emergency.pth')
        raise e

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(history['total'], label='Total Loss')
    plt.plot(history['bce'], label='BCE (Detection)')
    plt.yscale('log')
    plt.title('Training Curve - STGAT V2')
    plt.savefig('plots/training_curve_stgat_v2.png')

if __name__ == '__main__':
    train_stgat_v2()
