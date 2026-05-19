#!/usr/bin/env python3
"""
calculate_trimmed_mae_v9_5.py — METRICS QUARANTINE
==================================================
Global MAE vs Trimmed MAE (Excluding PLU, TNT, LWK, SRO).
"""

import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

_THIS_DIR = Path(__file__).parent
_ROOT_DIR = _THIS_DIR.parent
_V8_DIR   = _ROOT_DIR / "ScalogramV3_V8_Repository" / "model"
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_ROOT_DIR))
sys.path.insert(0, str(_V8_DIR))
sys.path.insert(0, r"d:\multi\scalogramv3")

from V9_5_Bayesian_Model import (
    MultiTaskScalogramV9_5_Bayesian,
    azimuth_deg_from_sincos,
)
from dataset_v3 import create_v94_dataloaders, STATION_MAP

def circular_mae_deg(pred_deg, true_deg):
    diff = np.abs(pred_deg - true_deg)
    diff = np.where(diff > 180.0, 360.0 - diff, diff)
    return diff

def calculate_metrics(ckpt_path, h5_path, priors_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader = create_v94_dataloaders(h5_path, batch_size=16, priors_dir=priors_dir)
    
    model = MultiTaskScalogramV9_5_Bayesian().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    id_to_stn = {v: k for k, v in STATION_MAP.items()}
    anomalous_stations = ['PLU', 'TNT', 'LWK', 'SRO']
    
    all_errors = []
    trimmed_errors = []
    
    print(f"Starting Inference on Validation Set...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            x_img, x_cosmic, prior_vec, stn_id, y_event, y_mag, y_azm = batch
            
            mask = (y_event == 1)
            if mask.sum() < 1: continue
            
            x_img, x_cosmic, prior_vec, stn_id = x_img.to(device), x_cosmic.to(device), prior_vec.to(device), stn_id.to(device)
            _, _, out_azm, _ = model(x_img, x_cosmic, prior_vec, stn_id)
            
            pred_deg = azimuth_deg_from_sincos(out_azm).cpu().numpy()
            true_deg = y_azm.numpy()
            
            # Filter by mask
            m_idx = mask.numpy().astype(bool)
            batch_preds = pred_deg[m_idx]
            batch_trues = true_deg[m_idx]
            batch_stns  = stn_id[m_idx].cpu().numpy()
            
            for p, t, sid in zip(batch_preds, batch_trues, batch_stns):
                err = circular_mae_deg(p, t)
                all_errors.append(err)
                
                stn_name = id_to_stn.get(sid, "UNKNOWN")
                if stn_name not in anomalous_stations:
                    trimmed_errors.append(err)

    global_mae = np.mean(all_errors)
    trimmed_mae = np.mean(trimmed_errors)
    
    print("\n" + "="*50)
    print(" V9.5 FINAL METRICS (VALIDATION SET)")
    print("="*50)
    print(f"Total Samples    : {len(all_errors)}")
    print(f"Trimmed Samples  : {len(trimmed_errors)} (Excl: {anomalous_stations})")
    print("-" * 50)
    print(f"Global Val MAE   : {global_mae:.2f}°")
    print(f"Trimmed Val MAE  : {trimmed_mae:.2f}°")
    print("="*50)
    
    if trimmed_mae < 25.0:
        print("!!! TARGET ACHIEVED: Trimmed MAE < 25° !!!")
    else:
        print(f"Target Check: {trimmed_mae:.2f}° (Goal: < 25°)")

if __name__ == "__main__":
    # Updated paths for workspace d:\multi\1multisite
    CKPT = r"d:\multi\1multisite\1scalogram_check\V9.5_Champion_Conditional_Physics\models\v9_5_best.pth"
    H5   = r"d:\multi\scalogramv3\scalogram_v3_cosmic_final.h5"
    PRIORS = r"d:\multi\scalogramv3\Bayesian\priors"
    calculate_metrics(CKPT, H5, PRIORS)
