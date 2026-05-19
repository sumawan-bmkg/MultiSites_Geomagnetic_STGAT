#!/usr/bin/env python3
"""train_azimuth_v9_5.py — V9.5 CSMP TRAINING"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np

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

# Inverse map for logging
ID_TO_STATION = {v: k for k, v in STATION_MAP.items()}
ID_TO_STATION[0] = "UNKNOWN"

class VonMisesDirectionalLoss(nn.Module):
    def forward(self, pred, target_deg, mask=None):
        pred_unit = F.normalize(pred, p=2, dim=1)
        pred_rad = torch.atan2(pred_unit[:, 0], pred_unit[:, 1])
        target_rad = target_deg * (math.pi / 180.0)
        loss = 1.0 - torch.cos(pred_rad - target_rad)
        if mask is not None:
            return (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)
        return loss.mean()

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self._f = open(str(log_path), 'w', encoding='utf-8')
    def __call__(self, msg):
        print(msg)
        self._f.write(msg + '\n')
        self._f.flush()

def circular_mae_deg(pred_deg, true_deg):
    diff = (pred_deg - true_deg).abs()
    diff = torch.where(diff > 180.0, 360.0 - diff, diff)
    return diff

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    model._v8.eval()
    total_loss, total_mae, total_gate = 0, 0, 0
    n_batches = 0
    for batch in loader:
        x_img, x_cosmic, prior_vec, stn_id, y_event, y_mag, y_azm = batch
        x_img, x_cosmic, prior_vec, stn_id, y_azm = x_img.to(device), x_cosmic.to(device), prior_vec.to(device), stn_id.to(device), y_azm.to(device, dtype=torch.float32)
        mask = (y_event == 1).to(device)
        if mask.sum() < 1: continue
        
        optimizer.zero_grad()
        _, _, out_azimuth, gate_val = model(x_img, x_cosmic, prior_vec, stn_id)
        loss = criterion(out_azimuth, y_azm, mask=mask) + 0.5 * torch.mean((1.0 - gate_val)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            pred_deg = azimuth_deg_from_sincos(out_azimuth)
            maes = circular_mae_deg(pred_deg[mask.bool()], y_azm[mask.bool()])
            total_mae += maes.mean().item()
            total_loss += loss.item()
            total_gate += gate_val.item()
        n_batches += 1
    return total_loss/n_batches, total_mae/n_batches, total_gate/n_batches

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_mae, total_gate = 0, 0
    stn_errors = {stn: [] for stn in ID_TO_STATION.values()}
    n_batches = 0
    for batch in loader:
        x_img, x_cosmic, prior_vec, stn_id, y_event, y_mag, y_azm = batch
        x_img, x_cosmic, prior_vec, stn_id, y_azm = x_img.to(device), x_cosmic.to(device), prior_vec.to(device), stn_id.to(device), y_azm.to(device, dtype=torch.float32)
        mask = (y_event == 1).to(device)
        if mask.sum() < 1: continue
        
        _, _, out_azimuth, gate_val = model(x_img, x_cosmic, prior_vec, stn_id)
        pred_deg = azimuth_deg_from_sincos(out_azimuth)
        
        # Per-station error tracking
        idx_anom = mask.bool()
        maes = circular_mae_deg(pred_deg[idx_anom], y_azm[idx_anom])
        stn_ids_batch = stn_id[idx_anom].cpu().numpy()
        for i, sid in enumerate(stn_ids_batch):
            stn_name = ID_TO_STATION.get(sid, "UNKNOWN")
            stn_errors[stn_name].append(maes[i].item())
            
        total_mae += maes.mean().item()
        total_gate += gate_val.item()
        n_batches += 1
    
    # Calculate per-station average
    avg_stn_errors = {k: np.mean(v) if v else 0.0 for k, v in stn_errors.items()}
    return total_mae/n_batches, total_gate/n_batches, avg_stn_errors

def train(args):
    out_dir = Path(args.output_dir)
    log = Logger(out_dir / "v9_5_training.log")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = create_v94_dataloaders(args.h5_path, args.batch_size, priors_dir=args.priors_dir)
    model = MultiTaskScalogramV9_5_Bayesian().to(DEVICE)
    model.load_v8_checkpoint(args.ckpt_v8)
    
    # High weight decay (0.1) for station-awareness regularization
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
    criterion = VonMisesDirectionalLoss()
    
    best_val_mae = float('inf')
    for epoch in range(1, args.epochs + 1):
        t_loss, t_mae, t_gate = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        v_mae, v_gate, stn_errs = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        log(f"Epoch {epoch:02d} | Loss: {t_loss:.4f} | Train MAE: {t_mae:.2f} | Val MAE: {v_mae:.2f} | Gate V: {v_gate:.3f}")
        
        # Periodic station error logging
        if epoch % 2 == 0:
            top_3_bad = sorted(stn_errs.items(), key=lambda x: x[1], reverse=True)[:3]
            log(f"  [Diagnostics] Top 3 Bad Stations: {top_3_bad}")
            
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            torch.save(model.state_dict(), out_dir / "v9_5_best.pth")
            log(f"  --> NEW BEST Val MAE: {best_val_mae:.2f}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5-path", default=str(Path(r"d:\\multi\\scalogramv3\\scalogram_v3_cosmic_final.h5")))
    p.add_argument("--ckpt-v8", default=str(Path(r"d:\\multi\\scalogramv3\\checkpoints\\v3_v8_conv_fpr_best_weights.pth")))
    p.add_argument("--priors-dir", default=str(Path(r"d:\\multi\\scalogramv3\\Bayesian\\priors")))
    p.add_argument("--output-dir", default=str(Path(r"d:\\multi\\scalogramv3\\Bayesian")))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
