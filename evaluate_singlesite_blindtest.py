import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import h5py
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

# ── Import V9.5 Architecture ────────────────────────────────────────────────
# We need to point to both Bayesian and ScalogramV3_V8_Repository/model
ROOT_DIR = Path(r"d:\multi")
BAYESIAN_DIR = ROOT_DIR / "scalogramv3" / "Bayesian"
V8_MODEL_DIR = ROOT_DIR / "scalogramv3" / "ScalogramV3_V8_Repository" / "model"

sys.path.insert(0, str(BAYESIAN_DIR))
sys.path.insert(0, str(V8_MODEL_DIR))

from V9_5_Bayesian_Model import MultiTaskScalogramV9_5_Bayesian
from dataset_v3 import STATION_MAP

# ── Configuration ──────────────────────────────────────────────────────────
H5_PATH = 'dataset_v13_blindtest_M5_patched.h5'
CKPT_PATH = BAYESIAN_DIR / 'v9_5_best.pth'
PRIORS_DIR = BAYESIAN_DIR / 'priors'

def evaluate_v95_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating V9.5 Champion Baseline on {device}...")

    # 1. Load Model
    model = MultiTaskScalogramV9_5_Bayesian().to(device)
    if not os.path.exists(CKPT_PATH):
        print(f"ERROR: Checkpoint {CKPT_PATH} not found!")
        return
    
    # Load state dict
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # 2. Prepare Station Priors Cache
    prior_cache = {}
    def get_prior(stn_name):
        if stn_name in prior_cache:
            return prior_cache[stn_name]
        p_path = PRIORS_DIR / f"prior_{stn_name}.pt"
        if p_path.exists():
            p = torch.load(p_path, map_location='cpu')
        else:
            p = torch.ones(360) / 360.0 # Uniform default
        prior_cache[stn_name] = p.to(device).unsqueeze(0)
        return prior_cache[stn_name]

    # 3. Process Blind Test
    with h5py.File(H5_PATH, 'r') as f:
        grp = f['test']
        dates = [d.decode() if isinstance(d, bytes) else d for d in grp['dates']]
        stations = [s.decode() if isinstance(s, bytes) else s for s in grp.attrs['stations']]
        n_days = len(dates)
        
        y_true_all = []
        daily_probs_all = []
        results = []

        print(f"Iterating {n_days} days in 2026...")
        with torch.no_grad():
            for i in tqdm(range(n_days)):
                date_str = dates[i]
                y_event = grp['label_event'][i]
                tensors = grp['tensors'][i] # (24, 3, 128, 1440)
                cosmic_raw = grp['cosmic_features'][i] # (Kp, Dst)
                
                # Normalization
                kp_norm = cosmic_raw[0] / 9.0
                dst_norm = np.tanh(cosmic_raw[1] / 50.0)
                x_cosmic = torch.tensor([kp_norm, dst_norm], device=device).float().unsqueeze(0)
                
                day_stn_probs = []
                best_stn = "NONE"
                max_prob = -1.0

                for s_idx, stn_name in enumerate(stations):
                    stn_tensor = torch.from_numpy(tensors[s_idx]).float().to(device).unsqueeze(0)
                    
                    # Skip empty stations
                    if torch.sum(torch.abs(stn_tensor)) < 1e-6:
                        continue
                    
                    # [EXPERIMENT] EXTREME GEOPHYSICS ABLATION
                    ablated_tensor = stn_tensor.clone()
                    # 1. Z-CHANNEL AMPUTATION (Channel index 2)
                    ablated_tensor[:, 2, :, :] = 0.0
                    # 2. SPECTRAL MUTILATION (Zeroing low freqs 0-32)
                    ablated_tensor[:, :, 0:32, :] = 0.0
                    
                    stn_tensor = ablated_tensor
                    
                    stn_id_val = STATION_MAP.get(stn_name, 0)
                    stn_id = torch.tensor([stn_id_val], device=device, dtype=torch.long)
                    prior_vec = get_prior(stn_name)
                    
                    # Forward pass
                    # out_detection shape is (1, 2)
                    out_detection, _, _, _ = model(stn_tensor, x_cosmic, prior_vec, stn_id)
                    
                    # Probability of class 1 (Earthquake)
                    prob = F.softmax(out_detection, dim=1)[0, 1].item()
                    
                    day_stn_probs.append(prob)
                    if prob > max_prob:
                        max_prob = prob
                        best_stn = stn_name
                
                # Max Pooling Logic
                daily_prob = max(day_stn_probs) if day_stn_probs else 0.0
                y_true_all.append(y_event)
                daily_probs_all.append(daily_prob)
                
                results.append({
                    'Date': date_str,
                    'Actual_Event': int(y_event),
                    'Max_Station_Prob': daily_prob,
                    'Station_ID_with_Max_Prob': best_stn
                })

    # 4. Calculate Metrics
    y_true = np.array(y_true_all)
    y_prob = np.array(daily_probs_all)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # 5. Report
    print("\n" + "="*50)
    print("=== CHAMPION V9.5 SINGLE-SITE BASELINE (2026 BLIND TEST) ===")
    print(f"Total Days Evaluated : {len(y_true)}")
    print(f"ROC-AUC Score        : {auc:.4f}")
    print(f"TPR (Sensitivity)    : {tpr:.2%}")
    print(f"FPR (False Alarm)    : {fpr:.2%}")
    print("="*50)

    # 6. Analysis Compass
    print("\n--- Analysis Compass ---")
    if auc < 0.55:
        print("RESULT: Terbukti Temporal Domain Shift (Sinyal alam 2026 berbeda dari 2018-2023).")
    elif auc > 0.65:
        print("RESULT: Terbukti STGAT V4 mengalami Noise Dilution (Fusi Multi-Site perlu diperbaiki).")
    else:
        print("RESULT: Ambiguous (Skor di zona transisi).")

    # 7. Save results
    os.makedirs('doc', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('doc/2026_blindtest_v9_5_baseline.csv', index=False)
    print(f"\nPredictions saved to doc/2026_blindtest_v9_5_baseline.csv")

if __name__ == "__main__":
    evaluate_v95_baseline()
