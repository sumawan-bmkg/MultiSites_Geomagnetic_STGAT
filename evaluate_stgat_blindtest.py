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
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

# ============================================================================
# ARCHITECTURE (Must match train_stgat_v1.py)
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
        feat = self.spatial_encoder(x)
        feat = self.pool(feat).squeeze(2).permute(0, 2, 1)
        self.temporal_encoder.flatten_parameters()
        gru_out, _ = self.temporal_encoder(feat)
        v_img = torch.mean(gru_out, dim=1)
        v_img = v_img.view(B, N, 512)
        v_consensus, att_weights = self.gat_layer(v_img)
        gate = self.cosmic_gate(x_cosmic)
        v_fusion = v_consensus * gate
        out_detect = self.head_detect(v_fusion).squeeze(-1)
        out_mag = self.head_mag(v_fusion).squeeze(-1)
        out_azm = self.head_azm(v_fusion)
        return out_detect, out_mag, out_azm, self.alpha_regions

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
            self.stations = [s.decode() if isinstance(s, bytes) else s for s in f[split].attrs['stations']]
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
# EVALUATION PIPELINE
# ============================================================================

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}...")

    # Load Model
    model = ScalogramV4_STGAT(n_regions=6).to(device)
    CKPT_PATH = 'checkpoints/best_stgat_v1.pth'
    if not os.path.exists(CKPT_PATH):
        print(f"ERROR: Checkpoint {CKPT_PATH} not found!")
        return
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    os.makedirs('doc', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # 1. TAHAP 2: G5 SOLAR STORM STRESS TEST
    print("\n--- TAHAP 1: G5 Solar Storm Stress Test (Mei 2024) ---")
    val_dataset = SpatioTemporalDataset('dataset_v13_train_val_M5_patched.h5', split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    g5_preds = []
    g5_dates = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Testing G5 Storm Period"):
            x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, date = batch
            d_str = date[0]
            if '2024-05-10' <= d_str <= '2024-05-12':
                x_img, x_cosmic = x_img.to(device), x_cosmic.to(device)
                out_detect, _, _, _ = model(x_img, x_cosmic)
                prob = torch.sigmoid(out_detect).item()
                g5_preds.append(prob)
                g5_dates.append(d_str)

    g5_fpr = np.mean(np.array(g5_preds) >= 0.5) if g5_preds else 0
    print(f"G5 Storm Period Samples: {len(g5_preds)}")
    print(f"G5 False Positive Rate (FPR): {g5_fpr:.2%}")

    # 2. TAHAP 3: 2026 BLIND TEST EVALUATION
    print("\n--- TAHAP 2: 2026 Blind Test Evaluation ---")
    test_dataset = SpatioTemporalDataset('dataset_v13_blindtest_M5_patched.h5', split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    results = []
    y_true_all = []
    y_prob_all = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running Blind Test"):
            x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, date = batch
            x_img, x_cosmic = x_img.to(device), x_cosmic.to(device)
            out_detect, out_mag, out_azm, _ = model(x_img, x_cosmic)
            
            prob = torch.sigmoid(out_detect).item()
            pred_event = 1 if prob >= 0.5 else 0
            pred_mag = out_mag.item()
            
            # Azimuth conversion
            pred_azm_vec = out_azm.cpu().numpy()[0]
            pred_azm_rad = np.arctan2(pred_azm_vec[0], pred_azm_vec[1])
            pred_azm_deg = (np.degrees(pred_azm_rad) + 360) % 360
            
            y_true_all.append(y_event.item())
            y_prob_all.append(prob)
            
            # Metrics for TP
            spatial_error_km = 0
            azm_error_deg = 0
            mag_error = 0
            
            if y_event.item() == 1:
                # Actual Azimuth
                y_azm_deg = y_azm.item()
                azm_error_deg = abs(pred_azm_deg - y_azm_deg)
                if azm_error_deg > 180: azm_error_deg = 360 - azm_error_deg
                
                # Spatial Error (km)
                # Using nearest station distance as reference radius
                nearest_dist = torch.min(y_dist).item()
                spatial_error_km = nearest_dist * np.radians(azm_error_deg)
                mag_error = abs(pred_mag - y_mag.item())
            
            results.append({
                'date': date[0],
                'y_event': y_event.item(),
                'prob': prob,
                'pred_event': pred_event,
                'y_mag': y_mag.item(),
                'pred_mag': pred_mag,
                'y_azm': y_azm.item(),
                'pred_azm': pred_azm_deg,
                'mag_error': mag_error,
                'azm_error': azm_error_deg,
                'spatial_error_km': spatial_error_km
            })

    df = pd.DataFrame(results)
    
    # Classification Metrics
    tn, fp, fn, tp = confusion_matrix(df['y_event'], df['pred_event']).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc = roc_auc_score(y_true_all, y_prob_all)
    
    # Physics Metrics (on TP only)
    tp_mask = (df['y_event'] == 1) & (df['pred_event'] == 1)
    df_tp = df[tp_mask]
    avg_mae_mag = df_tp['mag_error'].mean() if not df_tp.empty else 0
    avg_mae_azm = df_tp['azm_error'].mean() if not df_tp.empty else 0
    avg_spatial_error = df_tp['spatial_error_km'].mean() if not df_tp.empty else 0
    
    print(f"\nBlind Test Summary:")
    print(f"  TPR (Sensitivity): {tpr:.2%}")
    print(f"  FPR (False Alarm): {fpr:.2%}")
    print(f"  AUC Score: {auc:.4f}")
    print(f"  MAE Magnitude: {avg_mae_mag:.3f}")
    print(f"  MAE Azimuth: {avg_mae_azm:.2f} deg")
    print(f"  Avg Spatial Error: {avg_spatial_error:.2f} km")

    # Save CSV
    significant_events = df[df['y_event'] == 1][['date', 'y_mag', 'pred_mag', 'spatial_error_km']]
    significant_events.to_csv('doc/2026_blindtest_results.csv', index=False)
    print("CSV saved to doc/2026_blindtest_results.csv")

    # Plots
    plt.figure(figsize=(12, 6))
    
    # Plot 1: G5 Storm Classification
    plt.subplot(1, 2, 1)
    plt.plot(g5_dates, g5_preds, 'r-o', label='Detection Prob')
    plt.axhline(0.5, color='k', linestyle='--', label='Threshold')
    plt.xticks(rotation=45)
    plt.title('G5 Solar Storm Stress Test (May 2024)')
    plt.ylabel('Probability')
    plt.legend()
    
    # Plot 2: Regression Scatter
    plt.subplot(1, 2, 2)
    if not df_tp.empty:
        plt.scatter(df_tp['y_mag'], df_tp['pred_mag'], alpha=0.6)
        plt.plot([5, 9], [5, 9], 'k--')
        plt.xlabel('Actual Magnitude (Mw)')
        plt.ylabel('Predicted Magnitude (Mw)')
        plt.title('Blind Test 2026: Magnitude Regression')
    else:
        plt.text(0.5, 0.5, "No TPs for Regression Plot", ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/final_evaluation_results.png')
    print("Plot saved to plots/final_evaluation_results.png")

if __name__ == "__main__":
    evaluate()
