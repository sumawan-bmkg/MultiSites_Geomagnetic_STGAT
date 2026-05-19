#!/usr/bin/env python3
"""
train_azimuth_v9_1.py — V9.1: STATION-SPECIFIC DYNAMIC PRIORS TRAINING
======================================================================
Diagnosis V9.0:
  - Train MAE ~22° (model learns) BUT Val MAE ~78° (catastrophic mismatch)
  - Gate ≈ 0.005 → model 99.5% trusts Prior, but Prior was GLOBAL & WRONG
  - Azimuth is RELATIVE to station location → one-size-fits-all prior is invalid

Cure (V9.1):
  - Replace static global prior with station-specific KDE priors
  - Each sample gets its own prior based on station_id extracted from HDF5 metadata
  - Architecture stays IDENTICAL (Gated Von Mises, Frozen Backbone)
  - Only the PRIOR INPUT changes → the model can now trust prior correctly

Target: Val MAE < 25°
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

sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_ROOT_DIR))
sys.path.insert(0, str(_V8_DIR))

from V9_Bayesian_Model import (
    MultiTaskScalogramV9_Bayesian,
    build_v9_from_v8,
    azimuth_deg_from_sincos,
)
from dataset_v2 import (
    GeomagneticCosmicDatasetV2,
    create_v91_dataloaders,
)

# ─────────────────────────────────────────────────────────────────────────────
# Von Mises Directional Loss (identical to V9.0 — proven effective)
# ─────────────────────────────────────────────────────────────────────────────
class VonMisesDirectionalLoss(nn.Module):
    """Negative Log-Likelihood approximation of Von Mises distribution.
    Loss = 1 - cos(pred_angle - target_angle)
    Range: [0, 2], where 0 = perfect, 2 = opposite direction
    """
    def forward(self, pred: torch.Tensor, target_deg: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_unit = F.normalize(pred, p=2, dim=1)
        pred_rad = torch.atan2(pred_unit[:, 0], pred_unit[:, 1])
        target_rad = target_deg * (math.pi / 180.0)
        loss = 1.0 - torch.cos(pred_rad - target_rad)
        if mask is not None:
            mask_f = mask.float()
            denom = mask_f.sum().clamp(min=1e-8)
            return (loss * mask_f).sum() / denom
        return loss.mean()

# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.1):
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
    """Circular Mean Absolute Error in degrees."""
    diff = (pred_deg - true_deg).abs()
    diff = torch.where(diff > 180.0, 360.0 - diff, diff)
    return diff.mean().item()

def circular_median_ae_deg(pred_deg: torch.Tensor, true_deg: torch.Tensor) -> float:
    """Circular Median Absolute Error in degrees (robust metric)."""
    diff = (pred_deg - true_deg).abs()
    diff = torch.where(diff > 180.0, 360.0 - diff, diff)
    return diff.median().item()

class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(str(log_path), 'w', encoding='utf-8')
        self._history = []
        self._print_and_log(f"\n{'-'*70}")
        self._print_and_log(f"  V9.1 STATION-SPECIFIC DYNAMIC PRIORS — TRAINING LOG")
        self._print_and_log(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._print_and_log(f"{'-'*70}")

    def _print_and_log(self, msg: str):
        # Safe printing for Windows consoles that may not support Unicode.
        try:
            print(msg)
        except UnicodeEncodeError:
            sys.stdout.buffer.write((msg + "\n").encode('utf-8', errors='replace'))
        self._f.write(msg + '\n')
        self._f.flush()

    def __call__(self, msg: str):
        self._print_and_log(msg)

    def add_history(self, row: dict):
        self._history.append(row)

    def save_history_csv(self, csv_path: Path):
        if not self._history:
            return
        import csv
        keys = self._history[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._history)
        self._print_and_log(f"[OK] Training history → {csv_path}")

    def close(self):
        self._f.close()

# ─────────────────────────────────────────────────────────────────────────────
# Training Loop — ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device,
                    use_mask, gate_lambda=0.01) -> dict:
    model.train()
    model._v8.eval()  # Backbone ALWAYS in eval mode (frozen BN etc.)
    total_loss = 0.0
    total_gate = 0.0
    all_pred, all_true = [], []
    n_batches = 0
    for batch in loader:
        # Batch order: img, cosmic, prior, event, mag, azm
        x_img, x_cosmic, prior_vec, y_event, y_mag, y_azm = batch
        x_img = x_img.to(device)
        x_cosmic = x_cosmic.to(device)
        prior_vec = prior_vec.to(device)
        y_azm = y_azm.to(device, dtype=torch.float32)
        y_event = y_event.to(device)
        mask = (y_event == 1).float() if use_mask else None
        if mask is not None and mask.sum() < 1:
            continue
        optimizer.zero_grad(set_to_none=True)
        (_, _, out_azimuth, _, _, _, gate_val) = model(x_img, x_cosmic, prior_vec)
        dir_loss = criterion(out_azimuth, y_azm, mask=mask)
        loss = dir_loss + gate_lambda * gate_val
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head_azimuth_bayesian.parameters(), max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            pred_deg = azimuth_deg_from_sincos(out_azimuth)
            if mask is not None:
                idx = mask.bool()
                all_pred.append(pred_deg[idx].cpu())
                all_true.append(y_azm[idx].cpu())
            else:
                all_pred.append(pred_deg.cpu())
                all_true.append(y_azm.cpu())
        total_loss += dir_loss.item()
        total_gate += gate_val.item()
        n_batches += 1
    if all_pred:
        cat_pred = torch.cat(all_pred)
        cat_true = torch.cat(all_true)
        mae = circular_mae_deg(cat_pred, cat_true)
        median_ae = circular_median_ae_deg(cat_pred, cat_true)
    else:
        mae = median_ae = float('nan')
    return {
        'loss': total_loss / max(n_batches, 1),
        'mae_deg': mae,
        'median_ae_deg': median_ae,
        'avg_gate': total_gate / max(n_batches, 1),
        'n_samples': sum(len(p) for p in all_pred),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Validation Loop
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, use_mask) -> dict:
    model.eval()
    total_loss = 0.0
    total_gate = 0.0
    all_pred, all_true = [], []
    n_batches = 0
    for batch in loader:
        x_img, x_cosmic, prior_vec, y_event, y_mag, y_azm = batch
        x_img = x_img.to(device)
        x_cosmic = x_cosmic.to(device)
        prior_vec = prior_vec.to(device)
        y_azm = y_azm.to(device, dtype=torch.float32)
        y_event = y_event.to(device)
        mask = (y_event == 1).float() if use_mask else None
        (_, _, out_azimuth, _, _, _, gate_val) = model(x_img, x_cosmic, prior_vec)
        loss = criterion(out_azimuth, y_azm, mask=mask)
        if not torch.isnan(loss):
            total_loss += loss.item()
            total_gate += gate_val.item()
            n_batches += 1
        pred_deg = azimuth_deg_from_sincos(out_azimuth)
        if mask is not None:
            idx = mask.bool()
            all_pred.append(pred_deg[idx].cpu())
            all_true.append(y_azm[idx].cpu())
        else:
            all_pred.append(pred_deg.cpu())
            all_true.append(y_azm.cpu())
    if all_pred:
        cat_pred = torch.cat(all_pred)
        cat_true = torch.cat(all_true)
        mae = circular_mae_deg(cat_pred, cat_true)
        median_ae = circular_median_ae_deg(cat_pred, cat_true)
    else:
        mae = median_ae = float('nan')
    return {
        'val_loss': total_loss / max(n_batches, 1),
        'val_mae_deg': mae,
        'val_median_ae_deg': median_ae,
        'val_gate': total_gate / max(n_batches, 1),
        'n_samples': sum(len(p) for p in all_pred),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main Training Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "v9_1_training.log")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device          : {DEVICE}")
    log(f"H5 Path         : {args.h5_path}")
    log(f"V8 Checkpoint   : {args.ckpt_v8}")
    log(f"Priors Dir      : {args.priors_dir}")
    log(f"Epochs          : {args.epochs}")
    log(f"LR              : {args.lr}")
    log(f"Weight Decay    : {args.weight_decay}")
    log(f"Gate Lambda     : {args.gate_lambda}")
    log(f"Batch Size      : {args.batch_size}")
    log(f"Mask Positives  : {args.mask_positives}")
    log(f"Early Stop Pat. : {args.patience}")
    # Step 1: Data
    log(f"\n{'-'*70}")
    log(f"  STEP 1: Loading Dataset with Station-Specific Dynamic Priors")
    log(f"{'-'*70}")
    train_loader, val_loader = create_v91_dataloaders(
        h5_path=args.h5_path,
        batch_size=args.batch_size,
        num_workers=0,
        use_sampler=True,
        priors_dir=args.priors_dir,
    )
    log(f"[OK] Train batches: {len(train_loader)}")
    log(f"[OK] Val batches  : {len(val_loader)}")
    # Smoke test
    for batch in train_loader:
        x_img, x_cos, prior, y_ev, y_mg, y_az = batch
        log(f"[OK] Batch shapes: img={x_img.shape}, cosmic={x_cos.shape}, prior={prior.shape}, event={y_ev.shape}")
        log(f"[OK] Prior stats: sum={prior[0].sum():.4f}, max={prior[0].max():.4f}")
        if prior.shape[0] > 1:
            diff = (prior[0] - prior[1]).abs().sum().item()
            if diff > 0.001:
                log(f"[OK] ✅ DYNAMIC PRIORS VERIFIED: samples have DIFFERENT priors (diff={diff:.4f})")
            else:
                log(f"[WARN] ⚠️ First two samples have similar priors — may be same station")
        break
    # Step 2: Model
    log(f"\n{'-'*70}")
    log(f"  STEP 2: Building V9 Gated Bayesian Model (Frozen Backbone)")
    log(f"{'-'*70}")
    model = build_v9_from_v8(args.ckpt_v8, prior_dim=360, device=DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    log(f"[OK] Total params    : {total_params:,}")
    log(f"[OK] Trainable params: {trainable_params:,}")
    log(f"[OK] Frozen params   : {frozen_params:,}")
    log(f"\n[INFO] Trainable parameter groups:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            log(f"  {name}  ({param.numel():,} params)")
    # Step 3: Optimizer & Scheduler
    log(f"\n{'-'*70}")
    log(f"  STEP 3: Optimizer & Scheduler Setup")
    log(f"{'-'*70}")
    criterion = VonMisesDirectionalLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.1)
    log(f"[OK] Loss      : VonMisesDirectionalLoss")
    log(f"[OK] Optimizer : AdamW (lr={args.lr}, wd={args.weight_decay})")
    log(f"[OK] Scheduler : CosineAnnealingWarmRestarts (T_0=5, T_mult=1)")
    # Training loop
    log(f"\n{'='*70}")
    log(f"  STARTING V9.1 TRAINING — STATION-SPECIFIC DYNAMIC PRIORS")
    log(f"  Target: Val MAE < 25°")
    log(f"{'='*70}\n")
    best_val_mae = float('inf')
    ckpt_path = out_dir / "v9_1_dynamic_prior_best.pth"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE,
                                         use_mask=args.mask_positives,
                                         gate_lambda=args.gate_lambda)
        val_metrics = validate(model, val_loader, criterion, DEVICE, use_mask=args.mask_positives)
        lr_current = optimizer.param_groups[0]['lr']
        scheduler.step()
        elapsed = time.time() - t0
        log(f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss={train_metrics['loss']:.4f} MAE={train_metrics['mae_deg']:.2f}° "
            f"MedAE={train_metrics['median_ae_deg']:.2f}° | "
            f"ValMAE={val_metrics['val_mae_deg']:.2f}° "
            f"ValMedAE={val_metrics['val_median_ae_deg']:.2f}° | "
            f"Gate(T/V)={train_metrics['avg_gate']:.3f}/{val_metrics['val_gate']:.3f} | "
            f"LR={lr_current:.2e} | "
            f"Samples(T/V)={train_metrics['n_samples']}/{val_metrics['n_samples']} | "
            f"{elapsed:.1f}s")
        log.add_history({
            'epoch': epoch,
            'train_loss': f"{train_metrics['loss']:.4f}",
            'train_mae': f"{train_metrics['mae_deg']:.2f}",
            'train_median_ae': f"{train_metrics['median_ae_deg']:.2f}",
            'val_mae': f"{val_metrics['val_mae_deg']:.2f}",
            'val_median_ae': f"{val_metrics['val_median_ae_deg']:.2f}",
            'train_gate': f"{train_metrics['avg_gate']:.4f}",
            'val_gate': f"{val_metrics['val_gate']:.4f}",
            'lr': f"{lr_current:.2e}",
            'time_s': f"{elapsed:.1f}",
        })
        if val_metrics['val_mae_deg'] < best_val_mae:
            best_val_mae = val_metrics['val_mae_deg']
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_mae': best_val_mae,
                'val_median_ae': val_metrics['val_median_ae_deg'],
                'epoch': epoch,
                'train_mae': train_metrics['mae_deg'],
                'gate_val': val_metrics['val_gate'],
                'version': 'V9.1_dynamic_priors',
            }, str(ckpt_path))
            log(f"  ★ NEW BEST Val MAE = {best_val_mae:.2f}° → saved: {ckpt_path.name}")
            if best_val_mae < 25.0:
                log(f"  🎯 TARGET ACHIEVED! Val MAE = {best_val_mae:.2f}° < 25°")
        early_stopping(val_metrics['val_mae_deg'])
        if early_stopping.early_stop:
            log(f"  [STOP] Early stopping triggered at epoch {epoch}")
            break
    # Final report
    log(f"\n{'='*70}")
    log(f"  V9.1 TRAINING COMPLETE — STATION-SPECIFIC DYNAMIC PRIORS")
    log(f"{'='*70}")
    log(f"  Best Epoch       : {epoch}")
    log(f"  Best Val MAE     : {best_val_mae:.2f}°")
    log(f"  Checkpoint       : {ckpt_path}")
    target_status = "✅ ACHIEVED" if best_val_mae < 25.0 else "❌ NOT YET"
    log(f"  Target MAE < 25° : {target_status}")
    log(f"{'='*70}")
    log.save_history_csv(out_dir / "v9_1_training_history.csv")
    log.close()

# ─────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="V9.1: Train Gated Bayesian Azimuth with Station-Specific Dynamic Priors")
    p.add_argument("--h5-path", default=str(Path(r"d:\\multi\\scalogramv3\\scalogram_v3_cosmic_final.h5")))
    p.add_argument("--ckpt-v8", default=str(Path(r"d:\\multi\\scalogramv3\\checkpoints\\v3_v8_conv_fpr_best_weights.pth")))
    p.add_argument("--priors-dir", default=str(Path(r"d:\\multi\\scalogramv3\\Bayesian\\priors")))
    p.add_argument("--output-dir", default=str(Path(r"d:\\multi\\scalogramv3\\Bayesian")))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate (V9.0 used 2e-4, keeping same)")
    p.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for AdamW")
    p.add_argument("--gate-lambda", type=float, default=0.05, help="Gate penalty coefficient to prevent image dominance")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mask-positives", action='store_true', default=True, help="Only compute azimuth loss on positive (event=1) samples")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
