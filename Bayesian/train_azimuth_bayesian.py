#!/usr/bin/env python3
"""
train_azimuth_bayesian.py — MODULE 3: OPERATION "VON MISES GATING"
=============================================================
Strategy: Gated Fusion + Von Mises Directional Loss + Warm Restarts

This script implements an aggressive fine-tuning strategy for the V9 Gated 
Bayesian model to achieve high-precision azimuth prediction (Target MAE < 25°).
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ── Path setup ─────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent          # Bayesian/
_ROOT_DIR = _THIS_DIR.parent               # scalogramv3/
_V8_DIR   = _ROOT_DIR / "ScalogramV3_V8_Repository" / "model"

sys.path.insert(0, str(_ROOT_DIR))
sys.path.insert(0, str(_V8_DIR))

from V9_Bayesian_Model import (
    MultiTaskScalogramV9_Bayesian,
    build_v9_from_v8,
    azimuth_deg_from_sincos,
)
from V3_HDF5_DataLoader import GeomagneticCosmicDataset


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: Von Mises Directional Loss
# ─────────────────────────────────────────────────────────────────────────────
class VonMisesDirectionalLoss(nn.Module):
    """
    Negative Log-Likelihood approximation of Von Mises distribution.
    Strictly penalizes circular distance.
    """
    def forward(self, pred: torch.Tensor, target_deg: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Normalize output to unit circle
        pred_unit = F.normalize(pred, p=2, dim=1)
        
        # 2. Convert to radians
        # atan2(sin, cos) -> radians
        pred_rad = torch.atan2(pred_unit[:, 0], pred_unit[:, 1])
        target_rad = target_deg * (math.pi / 180.0)
        
        # 3. Circular Loss: 1 - cos(delta_theta)
        loss = 1.0 - torch.cos(pred_rad - target_rad)
        
        if mask is not None:
            mask_f = mask.float()
            denom = mask_f.sum().clamp(min=1e-8)
            return (loss * mask_f).sum() / denom
        
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping Helper
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_mae: float):
        if self.best_score is None:
            self.best_score = current_mae
        elif current_mae > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_mae
            self.counter = 0


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def circular_mae_deg(pred_deg: torch.Tensor, true_deg: torch.Tensor) -> float:
    diff = (pred_deg - true_deg).abs()
    diff = torch.where(diff > 180.0, 360.0 - diff, diff)
    return diff.mean().item()


class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(str(log_path), 'a', encoding='utf-8')
        self._print_and_log(f"\n{'='*60}")
        self._print_and_log(f"  V9 VON MISES GATING TRAINING LOG")
        self._print_and_log(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._print_and_log(f"{'='*60}")

    def _print_and_log(self, msg: str):
        print(msg)
        self._f.write(msg + '\n')
        self._f.flush()

    def __call__(self, msg: str):
        self._print_and_log(msg)

    def close(self):
        self._f.close()


class BayesianDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: GeomagneticCosmicDataset, prior_tensor: torch.Tensor):
        self.base = base_dataset
        self.prior = prior_tensor.cpu()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x_img, x_cosmic, y_event, y_mag, y_azm = self.base[idx]
        return x_img, x_cosmic, y_event, y_mag, y_azm, self.prior


# ─────────────────────────────────────────────────────────────────────────────
# Training Logic
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, use_mask, log, gate_lambda=0.01) -> dict:
    model.train()
    model._v8.eval()
    total_loss, total_gate, all_pred, all_true, n_batches = 0.0, 0.0, [], [], 0
    for batch in loader:
        x_img, x_cosmic, y_event, y_mag, y_azm, prior_vec = batch
        x_img, x_cosmic, prior_vec = x_img.to(device), x_cosmic.to(device), prior_vec.to(device)
        y_azm, y_event = y_azm.to(device, dtype=torch.float32), y_event.to(device)
        
        mask = (y_event == 1).float() if use_mask else None
        if mask is not None and mask.sum() < 1: continue

        optimizer.zero_grad(set_to_none=True)
        (_, _, out_azimuth, _, _, _, gate_val) = model(x_img, x_cosmic, prior_vec)
        
        # Directional Loss
        dir_loss = criterion(out_azimuth, y_azm, mask=mask)
        
        # [NEW] Gate Penalty Loss: Prevent Prior Collapse by penalizing high gate (image dominance)
        gate_penalty = gate_lambda * gate_val
        loss = dir_loss + gate_penalty
        
        if torch.isnan(loss) or torch.isinf(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head_azimuth_bayesian.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            pred_deg = azimuth_deg_from_sincos(out_azimuth)
            all_pred.append(pred_deg.cpu()); all_true.append(y_azm.cpu())
        total_loss += dir_loss.item(); total_gate += gate_val.item(); n_batches += 1

    mae = circular_mae_deg(torch.cat(all_pred), torch.cat(all_true)) if all_pred else float('nan')
    return {'loss': total_loss / max(n_batches, 1), 'mae_deg': mae, 'avg_gate': total_gate / max(n_batches, 1)}


@torch.no_grad()
def validate(model, loader, criterion, device, use_mask) -> dict:
    model.eval()
    total_loss, total_gate, all_pred, all_true, n_batches = 0.0, 0.0, [], [], 0
    for batch in loader:
        x_img, x_cosmic, y_event, y_mag, y_azm, prior_vec = batch
        x_img, x_cosmic, prior_vec = x_img.to(device), x_cosmic.to(device), prior_vec.to(device)
        y_azm, y_event = y_azm.to(device, dtype=torch.float32), y_event.to(device)
        mask = (y_event == 1).float() if use_mask else None
        (_, _, out_azimuth, _, _, _, gate_val) = model(x_img, x_cosmic, prior_vec)
        loss = criterion(out_azimuth, y_azm, mask=mask)
        if not torch.isnan(loss): total_loss += loss.item(); total_gate += gate_val.item(); n_batches += 1
        all_pred.append(azimuth_deg_from_sincos(out_azimuth).cpu()); all_true.append(y_azm.cpu())
    mae = circular_mae_deg(torch.cat(all_pred), torch.cat(all_true)) if all_pred else float('nan')
    return {'val_loss': total_loss / max(n_batches, 1), 'val_mae_deg': mae, 'val_gate': total_gate / max(n_batches, 1)}


def train(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "bayesian_training.log")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Prior
    prior_tensor = torch.load(args.prior_pt, map_location='cpu')
    prior_tensor /= prior_tensor.sum()
    
    # 2. Data
    base_train = GeomagneticCosmicDataset(args.h5_path, group_name='train')
    base_val = GeomagneticCosmicDataset(args.h5_path, group_name='val')
    train_loader = torch.utils.data.DataLoader(BayesianDataset(base_train, prior_tensor), batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(BayesianDataset(base_val, prior_tensor), batch_size=args.batch_size, shuffle=False)

    # 3. Model
    model = build_v9_from_v8(args.ckpt_v8, prior_dim=360, device=DEVICE)
    criterion = VonMisesDirectionalLoss()
    
    # TASK 3: Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    early_stopping = EarlyStopping(patience=7)

    log(f"Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    log(f"Scheduler: CosineAnnealingWarmRestarts (T_0=5)")

    best_val_mae = float('inf')
    ckpt_path = out_dir / "v9_gated_azimuth_best.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, args.mask_positives, log, gate_lambda=args.gate_lambda)
        val_metrics = validate(model, val_loader, criterion, DEVICE, args.mask_positives)
        
        # Check for warm restart (kick)
        lr_current = optimizer.param_groups[0]['lr']
        scheduler.step()
        elapsed = time.time() - t0
        
        log(f"Epoch {epoch:02d}/{args.epochs} | Loss={train_metrics['loss']:.4f} MAE={train_metrics['mae_deg']:.2f} deg | "
            f"ValMAE={val_metrics['val_mae_deg']:.2f} deg | Gate(T/V)={train_metrics['avg_gate']:.3f}/{val_metrics['val_gate']:.3f} | LR={lr_current:.2e} | {elapsed:.1f}s")

        if val_metrics['val_mae_deg'] < best_val_mae:
            best_val_mae = val_metrics['val_mae_deg']
            torch.save({'model_state_dict': model.state_dict(), 'val_mae': best_val_mae, 'epoch': epoch}, str(ckpt_path))
            log(f"  [NEW BEST] Val MAE = {best_val_mae:.2f} deg")

        early_stopping(val_metrics['val_mae_deg'])
        if early_stopping.early_stop:
            log(f"  [STOP] EARLY STOPPING triggered at epoch {epoch}")
            break

    log(f"\nTraining Complete. Best Val MAE: {best_val_mae:.2f} deg")
    log.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5-path", default=str(Path(r"d:\multi\scalogramv3\scalogram_v3_cosmic_final.h5")))
    p.add_argument("--ckpt-v8", default=str(Path(r"d:\multi\scalogramv3\checkpoints\v3_best_fusion_model.pth")))
    p.add_argument("--prior-pt", default=str(Path(r"d:\multi\scalogramv3\Bayesian\spatial_prior_1d.pt")))
    p.add_argument("--output-dir", default=str(Path(r"d:\multi\scalogramv3\Bayesian")))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-4) # Reduced from 5e-4
    p.add_argument("--weight-decay", type=float, default=0.05) # Increased from 0.01
    p.add_argument("--gate-lambda", type=float, default=0.05) # Penalty for image dominance
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mask-positives", action='store_true', default=True)
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
