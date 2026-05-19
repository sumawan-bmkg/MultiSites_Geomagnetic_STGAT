#!/usr/bin/env python3
"""
plot_xai_attention_v9_5.py — XAI: TEMPORAL ATTENTION VISUALIZATION
==================================================================
Tujuan: Membedah di mana model V9.5 menaruh perhatiannya pada stasiun anomali tinggi.
Output: PNG & PDF (Journal Ready Quality).
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set Journal Quality Styles
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 320,
})

_THIS_DIR = Path(__file__).parent
_ROOT_DIR = _THIS_DIR.parent
_V8_DIR   = _ROOT_DIR / "ScalogramV3_V8_Repository" / "model"
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_ROOT_DIR))
sys.path.insert(0, str(_V8_DIR))

from V9_5_Bayesian_Model import (
    MultiTaskScalogramV9_5_Bayesian,
    azimuth_deg_from_sincos,
)
from dataset_v3 import create_v94_dataloaders, STATION_MAP

# Global container for hook output
captured_attention = None

def attention_hook(module, input, output):
    global captured_attention
    captured_attention = output.detach().cpu().numpy()

def run_xai(ckpt_path, h5_path, priors_dir, target_stations=['PLU', 'TNT', 'LWK']):
    global captured_attention
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    _, val_loader = create_v94_dataloaders(h5_path, batch_size=1, priors_dir=priors_dir)
    
    # 2. Load Model
    model = MultiTaskScalogramV9_5_Bayesian().to(device)
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[OK] Model loaded: {ckpt_path}")
    else:
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    model.eval()
    
    # [TASK 1] Register Hook on the Sigmoid layer of attention_net
    # model.head_azimuth_bayesian.physics_encoder.attention_net[5] is the Sigmoid
    handle = model.head_azimuth_bayesian.physics_encoder.attention_net[5].register_forward_hook(attention_hook)

    # 3. Find Targeted Samples
    id_to_stn = {v: k for k, v in STATION_MAP.items()}
    samples_found = 0
    max_samples = 4 # We only need a few high-quality plots
    
    print(f"Scanning for stations: {target_stations}...")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x_img, x_cosmic, prior_vec, stn_id, y_event, y_mag, y_azm = batch
            
            stn_name = id_to_stn.get(stn_id.item(), "UNKNOWN")
            
            # Check if this sample is an anomaly and from a target station
            if y_event.item() == 1 and stn_name in target_stations:
                x_img, x_cosmic, prior_vec, stn_id = x_img.to(device), x_cosmic.to(device), prior_vec.to(device), stn_id.to(device)
                
                # Inference
                _, _, out_azm, _ = model(x_img, x_cosmic, prior_vec, stn_id)
                pred_deg = azimuth_deg_from_sincos(out_azm).item()
                true_deg = y_azm.item()
                err = abs(pred_deg - true_deg)
                if err > 180: err = 360 - err
                
                # Plotting [TASK 3]
                plot_xai_dashboard(x_img, captured_attention, stn_name, true_deg, pred_deg, err, samples_found)
                
                samples_found += 1
                if samples_found >= max_samples:
                    break

    handle.remove()
    print(f"XAI Audit Complete. {samples_found} plots generated.")

def plot_xai_dashboard(x_img, attn_weights, stn_name, true_deg, pred_deg, err, sample_idx):
    """Generates the dual-axis XAI plot."""
    # x_img: [1, 3, 128, 1440]
    # attn_weights: [1, 1, 1440]
    
    # 1. Extract Signal Energy (Z-channel = index 2)
    # Frequency mean absolute energy
    z_signal = torch.mean(torch.abs(x_img[0, 2]), dim=0).cpu().numpy() # [1440]
    attn = attn_weights[0, 0] # [1440]
    
    minutes = np.arange(1440)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Background: Raw Signal Energy
    color_signal = 'tab:grey'
    ax1.set_xlabel('Time Window (Minutes)')
    ax1.set_ylabel('Raw Z-Channel Energy (Scalogram Mean)', color=color_signal)
    ax1.plot(minutes, z_signal, color=color_signal, alpha=0.4, linewidth=0.8, label='Raw Signal (Z)')
    ax1.tick_params(axis='y', labelcolor=color_signal)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Foreground: Attention Weights
    ax2 = ax1.twinx()
    color_attn = 'tab:red' if err > 45 else 'tab:blue'
    ax2.set_ylabel('Model Attention Weight (0-1)', color=color_attn)
    ax2.plot(minutes, attn, color=color_attn, linewidth=1.5, label='Temporal Attention')
    ax2.fill_between(minutes, 0, attn, color=color_attn, alpha=0.15)
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelcolor=color_attn)

    # Annotations
    title = f"XAI Temporal Attention Map - Station: {stn_name}\n"
    title += f"True: {true_deg:.1f}°, Pred: {pred_deg:.1f}° (Error: {err:.1f}°)"
    plt.title(title, loc='left')
    
    fig.tight_layout()
    
    # Export [TASK 4]
    filename = f"xai_attention_{stn_name}_{sample_idx}"
    plt.savefig(f"{filename}.png", dpi=320, bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {filename}.png")

if __name__ == "__main__":
    CKPT = r"d:\multi\scalogramv3\Bayesian\v9_5_best.pth"
    H5   = r"d:\multi\scalogramv3\scalogram_v3_cosmic_final.h5"
    PRIORS = r"d:\multi\scalogramv3\Bayesian\priors"
    
    if not Path(CKPT).exists():
        print(f"Checkpoint not found at {CKPT}. Trying local dir...")
        CKPT = "v9_5_best.pth"
        
    run_xai(CKPT, H5, PRIORS)
