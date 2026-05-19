#!/usr/bin/env python3
"""
evaluate_mc_dropout_v9.py — EXTREME EVALUATION HACK
===================================================
Target: Memangkas Val MAE menggunakan Test-Time Bayesian Inference (MC Dropout).
N_SAMPLES = 50.
"""

import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

_THIS_DIR = Path(__file__).parent
_ROOT_DIR = _THIS_DIR.parent
_V8_DIR   = _ROOT_DIR / "ScalogramV3_V8_Repository" / "model"
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_ROOT_DIR))
sys.path.insert(0, str(_V8_DIR))

from V9_4_Bayesian_Model import (
    MultiTaskScalogramV9_4_Bayesian,
    azimuth_deg_from_sincos,
)
from dataset_v3 import create_v94_dataloaders

def circular_mae_deg_vectorized(pred_deg, true_deg):
    diff = (pred_deg - true_deg).abs()
    diff = torch.where(diff > 180.0, 360.0 - diff, diff)
    return diff.mean().item()

def enable_mc_dropout(model):
    """
    [TASK 1] SURGICAL DROPOUT ACTIVATION
    Aktifkan HANYA layer Dropout agar tetap berjalan meski model.eval() dipanggil.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            # print(f"[DEBUG] Activated MC Dropout for: {m}")

def run_evaluation_mc(ckpt_path, h5_path, priors_dir, n_samples=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Extreme Evaluation (MC Dropout) on: {device}")
    
    # 1. Load Data
    _, val_loader = create_v94_dataloaders(h5_path, batch_size=16, priors_dir=priors_dir)
    
    # 2. Load Model
    model = MultiTaskScalogramV9_4_Bayesian().to(device)
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return

    # --- MODE 1: STANDARD EVALUATION (Standard Inference) ---
    model.eval()
    std_maes = []
    
    print("\n--- Running Standard Evaluation ---")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Standard Pass"):
            x_img, x_cosmic, prior_vec, stn_id, y_event, y_mag, y_azm = batch
            x_img, x_cosmic, prior_vec, stn_id, y_azm = x_img.to(device), x_cosmic.to(device), prior_vec.to(device), stn_id.to(device), y_azm.to(device, dtype=torch.float32)
            
            mask = (y_event == 1).to(device)
            if mask.sum() < 1: continue
            
            _, _, out_azimuth, _ = model(x_img, x_cosmic, prior_vec, stn_id)
            pred_deg = azimuth_deg_from_sincos(out_azimuth)
            
            std_maes.append(circular_mae_deg_vectorized(pred_deg[mask], y_azm[mask]))
            
    std_mae_final = np.mean(std_maes)
    
    # --- MODE 2: MC DROPOUT EVALUATION (Bayesian Inference) ---
    # Set to eval for BatchNorm, then surgically enable Dropout
    model.eval()
    enable_mc_dropout(model)
    mc_maes = []
    
    print(f"\n--- Running MC Dropout Evaluation (N={n_samples}) ---")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="MC Dropout Passes"):
            x_img, x_cosmic, prior_vec, stn_id, y_event, y_mag, y_azm = batch
            x_img, x_cosmic, prior_vec, stn_id, y_azm = x_img.to(device), x_cosmic.to(device), prior_vec.to(device), stn_id.to(device), y_azm.to(device, dtype=torch.float32)
            
            mask = (y_event == 1).to(device)
            if mask.sum() < 1: continue
            
            # [TASK 2] N-FORWARD PASSES
            all_sin = []
            all_cos = []
            
            for _ in range(n_samples):
                _, _, out_azm_mc, _ = model(x_img, x_cosmic, prior_vec, stn_id)
                # out_azm_mc is [B, 2] -> [sin, cos] unit vector
                all_sin.append(out_azm_mc[:, 0]) # [B]
                all_cos.append(out_azm_mc[:, 1]) # [B]
                
            # [TASK 3] CIRCULAR MEAN AGGREGATION
            sin_stack = torch.stack(all_sin, dim=0) # [50, B]
            cos_stack = torch.stack(all_cos, dim=0) # [50, B]
            
            mean_sin = torch.mean(sin_stack, dim=0) # [B]
            mean_cos = torch.mean(cos_stack, dim=0) # [B]
            
            # Final MC Vector
            final_rad = torch.atan2(mean_sin, mean_cos) # [B]
            final_deg = torch.rad2deg(final_rad)
            final_deg = (final_deg + 360) % 360
            
            mc_maes.append(circular_mae_deg_vectorized(final_deg[mask], y_azm[mask]))

    mc_mae_final = np.mean(mc_maes)
    
    # --- [TASK 4] METRICS COMPARISON ---
    improvement = std_mae_final - mc_mae_final
    print("\n" + "="*60)
    print(" MC DROPOUT INFERENCE RESULTS (V9.4 ARCH)")
    print("="*60)
    print(f"Standard Val MAE   : {std_mae_final:.4f}°")
    print(f"MC Dropout Val MAE : {mc_mae_final:.4f}° (N={n_samples})")
    print(f"Absolute Improvement: {improvement:.4f}°")
    print("="*60)

if __name__ == "__main__":
    CKPT = r"d:\multi\scalogramv3\Bayesian\v9_4_best.pth"
    H5   = r"d:\multi\scalogramv3\scalogram_v3_cosmic_final.h5"
    PRIORS = r"d:\multi\scalogramv3\Bayesian\priors"
    
    run_evaluation_mc(CKPT, H5, PRIORS, n_samples=50)
