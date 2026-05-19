#!/usr/bin/env python3
"""
evaluate_mc_dropout_v9_optimized.py — EXTREME EVALUATION HACK (FAST)
==================================================================
Target: Memangkas Val MAE menggunakan Test-Time Bayesian Inference (MC Dropout).
OPTIMIZED: Pre-calculate backbone features.
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
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

@torch.no_grad()
def get_backbone_features(model, x_img, x_cosmic):
    """Replicates the backbone part of V9_4_Bayesian_Model.forward"""
    v8 = model._v8
    x = v8.features(x_img)
    x = v8.adaptive_pool(x)
    x = x.squeeze(2).permute(0, 2, 1)
    x = v8.gru_proj(x)
    v8.gru.flatten_parameters()
    gru_out, _ = v8.gru(x)
    v_img = torch.mean(gru_out, dim=1)
    
    station_features = v_img.unsqueeze(1).repeat(1, 8, 1)
    station_probs = torch.ones(v_img.shape[0], 8, device=v_img.device) * 0.5
    consensus_feat, _, _, _ = v8.gnn(station_features, station_probs)
    
    cosmic_attention = v8.cosmic_mlp(x_cosmic)
    v_fusion = consensus_feat * cosmic_attention
    return v_fusion

def run_evaluation_mc_optimized(ckpt_path, h5_path, priors_dir, n_samples=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Optimized Extreme Evaluation (MC Dropout) on: {device}")
    
    _, val_loader = create_v94_dataloaders(h5_path, batch_size=16, priors_dir=priors_dir)
    
    model = MultiTaskScalogramV9_4_Bayesian().to(device)
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return

    # --- MODE 1: STANDARD EVALUATION ---
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
    
    # --- MODE 2: MC DROPOUT EVALUATION (OPTIMIZED) ---
    model.eval()
    enable_mc_dropout(model)
    mc_maes = []
    
    print(f"\n--- Running Optimized MC Dropout Evaluation (N={n_samples}) ---")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="MC Dropout Inference"):
            x_img, x_cosmic, prior_vec, stn_id, y_event, y_mag, y_azm = batch
            x_img, x_cosmic, prior_vec, stn_id, y_azm = x_img.to(device), x_cosmic.to(device), prior_vec.to(device), stn_id.to(device), y_azm.to(device, dtype=torch.float32)
            
            mask = (y_event == 1).to(device)
            if mask.sum() < 1: continue
            
            # STEP A: Pre-calculate Backbone Features (Once per batch)
            v_fusion = get_backbone_features(model, x_img, x_cosmic)
            
            # STEP B: 50x Head Inference (Very Fast)
            all_sin = []
            all_cos = []
            
            azimuth_head = model.head_azimuth_bayesian
            for _ in range(n_samples):
                # out_azm_mc: [B, 2]
                out_azm_mc, _ = azimuth_head(v_fusion, prior_vec, x_img, stn_id)
                all_sin.append(out_azm_mc[:, 0])
                all_cos.append(out_azm_mc[:, 1])
                
            sin_stack = torch.stack(all_sin, dim=0)
            cos_stack = torch.stack(all_cos, dim=0)
            
            mean_sin = torch.mean(sin_stack, dim=0)
            mean_cos = torch.mean(cos_stack, dim=0)
            
            final_rad = torch.atan2(mean_sin, mean_cos)
            final_deg = torch.rad2deg(final_rad)
            final_deg = (final_deg + 360) % 360
            
            mc_maes.append(circular_mae_deg_vectorized(final_deg[mask], y_azm[mask]))

    mc_mae_final = np.mean(mc_maes)
    improvement = std_mae_final - mc_mae_final
    print("\n" + "="*60)
    print(" OPTIMIZED MC DROPOUT INFERENCE RESULTS (V9.4)")
    print("="*60)
    print(f"Standard Val MAE   : {std_mae_final:.4f}°")
    print(f"MC Dropout Val MAE : {mc_mae_final:.4f}° (N={n_samples})")
    print(f"Absolute Improvement: {improvement:.4f}°")
    print("="*60)

if __name__ == "__main__":
    CKPT = r"d:\multi\scalogramv3\Bayesian\v9_4_best.pth"
    H5   = r"d:\multi\scalogramv3\scalogram_v3_cosmic_final.h5"
    PRIORS = r"d:\multi\scalogramv3\Bayesian\priors"
    run_evaluation_mc_optimized(CKPT, H5, PRIORS, n_samples=50)
