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
from torch.utils.checkpoint import checkpoint

# ============================================================================
# ARCHITECTURE: ScalogramV4_STGAT (V2: Chunked Spatial Encoding)
# ============================================================================

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
    def __init__(self, n_regions=6):
        super().__init__()
        efficientnet = models.efficientnet_b1(weights=None)
        self.spatial_encoder = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.temporal_encoder = nn.GRU(
            input_size=1280, hidden_size=256, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        self.gat_layer = SpatialGNNModule(in_features=512, hidden=256, out_features=512)
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
        
        # V3: CHUNKED SPATIAL ENCODING + GRADIENT CHECKPOINTING
        if self.training:
            # Process N stations in chunks to save memory
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
# LOSS & DATASET (Same as V2)
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
        return total_loss, {'bce': loss_bce.item(), 'mag': loss_mag.item(), 'azm': loss_azm.item(), 'phys': loss_phys.item()}

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
            azms = g['label_azm'][idx]
            y_azm = torch.tensor(np.median(azms[azms > 0])).float() if np.any(azms > 0) else torch.tensor(0.0).float()
            y_dist = torch.from_numpy(g['label_dist'][idx]).float()
            if y_event == 1.0:
                nearest_idx = np.argmin(g['label_dist'][idx])
                region_idx = self.station_to_region[nearest_idx]
            else:
                region_idx = 0
        return tensor, cosmic, y_event, y_mag, y_azm, y_dist, region_idx

# ============================================================================
# RESUME PIPELINE: PHASE 2 ONLY
# ============================================================================

def resume_phase2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Resuming Phase 2 (Chunked) on device: {device}")
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    model = ScalogramV4_STGAT(n_regions=6).to(device)
    PHASE1_CKPT = 'checkpoints/stgat_v2_phase1_complete.pth'
    
    if os.path.exists(PHASE1_CKPT):
        print(f"Loading Phase 1 weights from {PHASE1_CKPT}...")
        model.load_state_dict(torch.load(PHASE1_CKPT, map_location=device))
    else:
        print(f"ERROR: {PHASE1_CKPT} not found. Cannot resume.")
        return

    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = PhysicsInformedLoss().to(device)

    dataset_path = 'dataset_v13_train_val_M5_patched.h5'
    train_loader = DataLoader(SpatioTemporalDataset(dataset_path, 'train'), batch_size=1, shuffle=True, num_workers=0, pin_memory=False)

    history = {'total': [], 'bce': [], 'mag': [], 'azm': [], 'phys': []}
    best_val_loss = float('inf')

    print("\nPhase 2 (Chunked Spatial Encoding) Dimulai...")
    try:
        for epoch in range(30):
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
                if (i + 1) % 10 == 0:
                    print(f"  Iter {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
                del loss, preds, x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, r_idx
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            print(f"Phase 2 Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
            torch.save(model.state_dict(), 'checkpoints/stgat_v2_resumed_last.pth')
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(model.state_dict(), 'checkpoints/best_stgat_v2.pth')
            gc.collect()

    except Exception as e:
        with open('logs/stgat_v2_resume_crash.txt', 'a') as f:
            f.write(f"\n[{datetime.now()}] RESUME CRASH\n")
            f.write(traceback.format_exc())
        torch.save(model.state_dict(), 'checkpoints/stgat_v2_resume_rescue.pth')
        raise e

if __name__ == '__main__':
    resume_phase2()
