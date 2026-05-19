#!/usr/bin/env python3
"""train_azimuth_v9_4.py — V9.4 ULTIMATE TRAINING"""

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

class VonMisesDirectionalLoss(nn.Module):
    def forward(self, pred, target_deg, mask=None):
        pred_unit = F.normalize(pred, p=2, dim=1)
        pred_rad = torch.atan2(pred_unit[:, 0], pred_unit[:, 1])
        target_rad = target_deg * (math.pi / 180.0)
        loss = 1.0 - torch.cos(pred_rad - target_rad)
        if mask is not None:
            mask_f = mask.float()
            denom = mask_f.sum().clamp(min=1e-8)
            return (loss * mask_f).sum() / denom
        return loss.mean()

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(str(log_path), 'w', encoding='utf-8')
    def __call__(self, msg):
        print(msg)
        self._f.write(msg + '\n')
        self._f.flush()

def circular_mae_deg(pred_deg, true_deg):
    diff = (pred_deg - true_deg).abs()
    diff = torch.where(diff > 180.0, 360.0 - diff, diff)
    return diff.mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device, gate_lambda=0.5):
    model.train()
    model._v8.eval()
    total_loss, total_mae, total_gate = 0, 0, 0
    n_batches = 0
    for batch in loader:
        x_img, x_cosmic, prior_vec, stn_id, y_event, y_mag, y_azm = batch
        x_img, x_cosmic, prior_vec, stn_id, y_azm = x_img.to(device), x_cosmic.to(device), prior_vec.to(device), stn_id.to(device), y_azm.to(device, dtype=torch.float32)
        mask = (y_event == 1).to(device).float()
        if mask.sum() < 1: continue
        
        optimizer.zero_grad()
        _, _, out_azimuth, gate_val = model(x_img, x_cosmic, prior_vec, stn_id)
        dir_loss = criterion(out_azimuth, y_azm, mask=mask)
        gate_penalty = torch.mean((1.0 - gate_val) ** 2)
        loss = dir_loss + gate_lambda * gate_penalty
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            pred_deg = azimuth_deg_from_sincos(out_azimuth)
            idx = mask.bool()
            mae = circular_mae_deg(pred_deg[idx], y_azm[idx])
            total_mae += mae
            total_loss += loss.item()
            total_gate += gate_val.item()
        n_batches += 1
    return total_loss/n_batches, total_mae/n_batches, total_gate/n_batches

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_mae, total_gate = 0, 0
    n_batches = 0
    for batch in loader:
        x_img, x_cosmic, prior_vec, stn_id, y_event, y_mag, y_azm = batch
        x_img, x_cosmic, prior_vec, stn_id, y_azm = x_img.to(device), x_cosmic.to(device), prior_vec.to(device), stn_id.to(device), y_azm.to(device, dtype=torch.float32)
        mask = (y_event == 1).to(device).float()
        if mask.sum() < 1: continue
        
        _, _, out_azimuth, gate_val = model(x_img, x_cosmic, prior_vec, stn_id)
        pred_deg = azimuth_deg_from_sincos(out_azimuth)
        idx = mask.bool()
        total_mae += circular_mae_deg(pred_deg[idx], y_azm[idx])
        total_gate += gate_val.item()
        n_batches += 1
    return total_mae/n_batches, total_gate/n_batches

def train(args):
    out_dir = Path(args.output_dir)
    log = Logger(out_dir / "v9_4_training.log")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = create_v94_dataloaders(args.h5_path, args.batch_size, priors_dir=args.priors_dir)
    model = MultiTaskScalogramV9_4_Bayesian().to(DEVICE)
    model.load_v8_checkpoint(args.ckpt_v8)
    
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
    criterion = VonMisesDirectionalLoss()
    
    best_val_mae = float('inf')
    for epoch in range(1, args.epochs + 1):
        t_loss, t_mae, t_gate = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        v_mae, v_gate = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        log(f"Epoch {epoch:02d} | Loss: {t_loss:.4f} | Train MAE: {t_mae:.2f} | Val MAE: {v_mae:.2f} | Gate V: {v_gate:.3f}")
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            torch.save(model.state_dict(), out_dir / "v9_4_best.pth")
            log(f"  --> New Best Val MAE: {best_val_mae:.2f}")

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
