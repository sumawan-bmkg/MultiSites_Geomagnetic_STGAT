import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

# ============================================================================
# TAHAP 1: DEFINISI DETERMINISTIC PHYSICS GATE
# ============================================================================

class PhysicsDeterministicGate(nn.Module):
    """
    Deterministic gate based on physics: 
    Kp Index < threshold -> Lolos (1.0)
    Kp Index > threshold -> Blokir (0.0)
    Uses a smooth sigmoid transition.
    """
    def __init__(self, kp_threshold=5.0):
        super(PhysicsDeterministicGate, self).__init__()
        self.kp_threshold = kp_threshold

    def forward(self, space_weather):
        # space_weather: [Batch, 2] -> [Kp, Dst]
        kp_index = space_weather[:, 0]
        
        # Sigmoid terbalik: 
        # Kp < threshold -> value > 0 -> sigmoid > 0.5
        # Kp > threshold -> value < 0 -> sigmoid < 0.5
        # Slope -2.0 for reasonably steep transition
        gate_val = torch.sigmoid(-2.0 * (kp_index - self.kp_threshold))
        
        return gate_val.unsqueeze(1) # [Batch, 1]

# ============================================================================
# TAHAP 2: MODIFIKASI ARSITEKTUR MODEL UNTUK INFERENSI
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
        
        # REPLACEMENT: Use Deterministic Physics Gate instead of MLP
        self.cosmic_gate = PhysicsDeterministicGate(kp_threshold=5.0)
        
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
        feat = self.spatial_encoder(x)
        feat = self.pool(feat).squeeze(2).permute(0, 2, 1)
        self.temporal_encoder.flatten_parameters()
        gru_out, _ = self.temporal_encoder(feat)
        v_img = torch.mean(gru_out, dim=1)
        v_img = v_img.view(B, N, 512)
        v_consensus, att_weights = self.gat_layer(v_img)
        
        # Detection Bypass (Raw)
        raw_pred_logits = self.head_detect(v_consensus).squeeze(-1)
        
        # Deterministic Gating
        gate_val = self.cosmic_gate(x_cosmic) # [B, 1]
        
        # Regression features (Gated)
        v_fusion = v_consensus * gate_val
        out_mag = self.head_mag(v_fusion).squeeze(-1)
        out_azm = self.head_azm(v_fusion)
        
        return raw_pred_logits, out_mag, out_azm, self.alpha_regions, gate_val

# ============================================================================
# DATASET
# ============================================================================

class SpatioTemporalDataset(Dataset):
    def __init__(self, h5_path, split='train'):
        self.h5_path = h5_path
        self.split = split
        with h5py.File(h5_path, 'r') as f:
            self.n_samples = len(f[split]['dates'])
            self.dates = [d.decode() if isinstance(d, bytes) else d for d in f[split]['dates']]
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
            date = self.dates[idx]
        return tensor, cosmic, y_event, y_mag, y_azm, y_dist, date

# ============================================================================
# TAHAP 3: EVALUASI MURNI & FISIKA DETERMINISTIK
# ============================================================================

def evaluate_v3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Deterministic Evaluation (V3) on {device}...")

    model = ScalogramV4_STGAT(n_regions=6).to(device)
    CKPT_PATH = 'checkpoints/best_stgat_v2.pth'
    if not os.path.exists(CKPT_PATH):
        print(f"ERROR: {CKPT_PATH} not found!")
        return
    
    # Load weights with strict=False (ignoring the cosmic_gate MLP params)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=False)
    model.eval()

    os.makedirs('doc', exist_ok=True)

    # 1. G5 SOLAR STORM STRESS TEST
    print("\n--- Testing G5 Storm Period (May 2024) ---")
    val_dataset = SpatioTemporalDataset('dataset_v13_train_val_M5_patched.h5', split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    g5_probs_raw = []
    g5_probs_gated = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="G5 Stress Test"):
            x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, date = batch
            d_str = date[0]
            if '2024-05-10' <= d_str <= '2024-05-12':
                x_img, x_cosmic = x_img.to(device), x_cosmic.to(device)
                raw_logits, _, _, _, gate_val = model(x_img, x_cosmic)
                
                prob_raw = torch.sigmoid(raw_logits).item()
                prob_gated = prob_raw * gate_val.item()
                
                g5_probs_raw.append(prob_raw)
                g5_probs_gated.append(prob_gated)

    g5_fpr_raw = np.mean(np.array(g5_probs_raw) >= 0.5) if g5_probs_raw else 0
    g5_fpr_gated = np.mean(np.array(g5_probs_gated) >= 0.5) if g5_probs_gated else 0

    # 2. 2026 BLIND TEST EVALUATION
    print("\n--- Testing 2026 Blind Test ---")
    test_dataset = SpatioTemporalDataset('dataset_v13_blindtest_M5_patched.h5', split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    results = []
    y_true_all = []
    probs_raw_all = []
    probs_gated_all = []
    gates_all = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Blind Test"):
            x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, date = batch
            x_img, x_cosmic = x_img.to(device), x_cosmic.to(device)
            raw_logits, out_mag, out_azm, _, gate_val = model(x_img, x_cosmic)
            
            prob_raw = torch.sigmoid(raw_logits).item()
            prob_gated = prob_raw * gate_val.item()
            
            y_true_all.append(y_event.item())
            probs_raw_all.append(prob_raw)
            probs_gated_all.append(prob_gated)
            gates_all.append(gate_val.item())
            
            results.append({
                'date': date[0],
                'y_event': y_event.item(),
                'prob_raw': prob_raw,
                'prob_gated': prob_gated,
                'gate_val': gate_val.item(),
                'y_mag': y_mag.item(),
                'pred_mag': out_mag.item()
            })

    # Metrics
    raw_auc = roc_auc_score(y_true_all, probs_raw_all)
    gated_auc = roc_auc_score(y_true_all, probs_gated_all)
    mean_gate = np.mean(gates_all)

    # OUTPUT LAPORAN
    print("\n" + "="*40)
    print("=== STGAT DETERMINISTIC EVALUATION ===")
    print("1. G5 Solar Storm Stress Test (May 2024)")
    print(f"   - FPR (Raw)   : {g5_fpr_raw:.2%}")
    print(f"   - FPR (Gated) : {g5_fpr_gated:.2%}")
    print("\n2. 2026 Blind Test (Raw vs Gated)")
    print(f"   - RAW AUC Score   : {raw_auc:.4f}")
    print(f"   - GATED AUC Score : {gated_auc:.4f}")
    print(f"   - Mean Gate Value (2026) : {mean_gate:.4f}")
    print("="*40)

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv('doc/2026_blindtest_deterministic.csv', index=False)
    print("Results saved to doc/2026_blindtest_deterministic.csv")

if __name__ == "__main__":
    evaluate_v3()
