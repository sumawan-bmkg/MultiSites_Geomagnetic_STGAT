#!/usr/bin/env python3
"""
V9_Bayesian_Model.py — MODULE 2: BAYESIAN GRAFTED AZIMUTH ARCHITECTURE (OVERHAULED)
=========================================================================
Strategy: Gated Multimodal Fusion (The Cure for Prior Collapse)

This version implements a sigmoid-based gating mechanism to balance 
ULF Scalogram image features with Physics-Informed Spatial Priors.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ─────────────────────────────────────────────────────────────
_THIS_DIR  = Path(__file__).parent                    # Bayesian/
_ROOT_DIR  = _THIS_DIR.parent                         # scalogramv3/
_V8_DIR    = _ROOT_DIR / "ScalogramV3_V8_Repository" / "model"

sys.path.insert(0, str(_ROOT_DIR))
sys.path.insert(0, str(_V8_DIR))

from V3_Model_v8 import MultiTaskScalogramV3_v8        # base V8


# ─────────────────────────────────────────────────────────────────────────────
# Gated Bayesian Azimuth Head
# ─────────────────────────────────────────────────────────────────────────────
class BayesianAzimuthHead(nn.Module):
    """
    Azimuth head with Gated Multimodal Fusion.
    Balances image features with spatial prior to prevent 'Prior Collapse'.

    Architecture:
    1. Prior Projection: 360-D -> 128-D
    2. Image Projection: 512-D -> 128-D (Matches user's 128-D intent)
    3. Gating Mechanism: sigmoid(Linear(256 -> 128))
    4. Fusion: (gate * img) + ((1 - gate) * prior)
    5. Output Head: Linear(128 -> 64) -> ReLU -> Linear(64 -> 2)
    """

    def __init__(self, img_dim: int = 512, prior_dim: int = 360):
        super().__init__()
        
        # 1. Prior features projection (360 -> 128)
        self.prior_proj = nn.Sequential(
            nn.Linear(prior_dim, 128),
            nn.LayerNorm(128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True)
        )
        
        # 2. Image features projection (512 -> 128)
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LayerNorm(128),
            nn.Dropout(p=0.2), # [NEW] Prevent image dominance
            nn.ReLU(inplace=True)
        )
        
        # 3. Gating mechanism (Attention)
        # Input: cat([img_128, prior_128]) -> 256
        self.gate_net = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.Sigmoid()
        )
        
        # 4. Output head
        self.out_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3), # [NEW] Regularize the unit vector output
            nn.Linear(64, 2)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # FAVOR PRIOR AT START: Initialize gate bias to -1.0 
                    # so gate = sigmoid(-1.0 + noise) ~ 0.27 (73% Prior trust)
                    if m == self.gate_net[0]:
                        nn.init.constant_(m.bias, -1.0)
                    else:
                        nn.init.constant_(m.bias, 0.0)

    def forward(self, img_feat: torch.Tensor,
                prior_vec: torch.Tensor) -> torch.Tensor:
        """
        Gated Fusion Forward Pass.
        """
        # Step 1: Feature Projection
        p_feat = self.prior_proj(prior_vec)  # (B, 128)
        i_feat = self.img_proj(img_feat)     # (B, 128)
        
        # Step 2: Gate Calculation
        combined = torch.cat([i_feat, p_feat], dim=-1) # (B, 256)
        gate = self.gate_net(combined)                 # (B, 128)
        
        # [NEW] Stochastic Gating (Dropout on image features during training)
        if self.training:
            # Randomly zero out image features for some samples to force prior reliance
            img_mask = (torch.rand(i_feat.size(0), 1, device=i_feat.device) > 0.1).float()
            i_feat = i_feat * img_mask
        
        # Step 3: Gated Multimodal Fusion
        # If gate -> 1, model trusts Image. If gate -> 0, model trusts Prior.
        fused = (gate * i_feat) + ((1.0 - gate) * p_feat) # (B, 128)
        
        # Step 4: Output Unit Vector
        raw_out = self.out_net(fused)                     # (B, 2)
        unit_vec = F.normalize(raw_out, p=2, dim=1)       # (B, 2)
        return unit_vec, gate.mean() # Return gate mean for monitoring


# ─────────────────────────────────────────────────────────────────────────────
# V9 Bayesian Model: V8 backbone (frozen) + Gated Bayesian Head (trainable)
# ─────────────────────────────────────────────────────────────────────────────
class MultiTaskScalogramV9_Bayesian(nn.Module):
    """
    ScalogramV9 — Physics-Informed Gated Bayesian Azimuth Model.
    """

    _FROZEN_GROUPS = [
        "features", "adaptive_pool", "gru_proj", "gru", 
        "gnn", "cosmic_mlp", "head_detection", "head_magnitude", 
        "head_azimuth", "projection_head"
    ]

    def __init__(self, prior_dim: int = 360, img_feat_dim: int = 512):
        super().__init__()
        self.prior_dim    = prior_dim
        self.img_feat_dim = img_feat_dim
        self._v8 = MultiTaskScalogramV3_v8(pretrained=False)
        self._freeze_v8_backbone()
        
        # Graft: New Gated Bayesian head
        self.head_azimuth_bayesian = BayesianAzimuthHead(
            img_dim=img_feat_dim,
            prior_dim=prior_dim,
        )

    def _freeze_v8_backbone(self):
        frozen_params = 0
        for group_name in self._FROZEN_GROUPS:
            module = getattr(self._v8, group_name, None)
            if module is None: continue
            for param in module.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        print(f"[OK] Frozen {frozen_params:,} parameters from V8 backbone.")

    def load_v8_checkpoint(self, ckpt_path: str) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            print(f"[WARN] Checkpoint not found: {path}")
            return
        ckpt = torch.load(str(path), map_location='cpu')
        sd = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
        sd_clean = {k.replace('module.', ''): v for k, v in sd.items()}
        self._v8.load_state_dict(sd_clean, strict=False)
        self._freeze_v8_backbone()

    def _extract_v8_features(self, x_img, x_cosmic, x_mask=None, T=1.0):
        v8 = self._v8
        x = v8.features(x_img)
        x = v8.adaptive_pool(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = v8.gru_proj(x)
        if x_mask is not None:
            tgt = x.size(1)
            m = (F.adaptive_avg_pool1d(x_mask.unsqueeze(1), tgt) > 0.5).float()
            x = x * m.permute(0, 2, 1)
        v8.gru.flatten_parameters()
        gru_out, _ = v8.gru(x)
        v_img = torch.mean(gru_out, dim=1)
        station_features = v_img.unsqueeze(1).repeat(1, 8, 1)
        station_probs = torch.ones(v_img.shape[0], 8, device=v_img.device) * 0.5
        consensus_feat, reg_score, _, att_weights = v8.gnn(station_features, station_probs)
        cosmic_attention = v8.cosmic_mlp(x_cosmic)
        v_fusion = consensus_feat * cosmic_attention
        out_detection = v8.head_detection(v_fusion) / T
        out_magnitude = v8.head_magnitude(v_fusion)
        proj_vec = F.normalize(v8.projection_head(v_fusion), p=2, dim=1)
        return v_fusion, out_detection, out_magnitude, reg_score, att_weights, proj_vec

    def forward(self, x_img, x_cosmic, prior_vec, x_mask=None, T=1.0):
        (v_fusion, out_detection, out_magnitude,
         reg_score, att_weights, proj_vec) = self._extract_v8_features(x_img, x_cosmic, x_mask, T)
        out_azimuth, gate_val = self.head_azimuth_bayesian(v_fusion, prior_vec)
        return out_detection, out_magnitude, out_azimuth, reg_score, att_weights, proj_vec, gate_val


def build_v9_from_v8(ckpt_path, prior_dim=360, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskScalogramV9_Bayesian(prior_dim=prior_dim).to(device)
    model.load_v8_checkpoint(ckpt_path)
    return model


def azimuth_deg_from_sincos(pred: torch.Tensor) -> torch.Tensor:
    unit = F.normalize(pred, p=2, dim=1)
    deg  = torch.atan2(unit[:, 0], unit[:, 1]) * (180.0 / torch.pi)
    return deg % 360.0


# Smoke test
if __name__ == "__main__":
    print("=" * 60)
    print("  V9 GATED BAYESIAN MODEL — SMOKE TEST")
    print("=" * 60)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskScalogramV9_Bayesian(prior_dim=360).to(DEVICE)
    B = 2
    dummy_img = torch.randn(B, 3, 128, 1440).to(DEVICE)
    dummy_cosmic = torch.randn(B, 2).to(DEVICE)
    prior_batch = (torch.ones(360) / 360.0).unsqueeze(0).repeat(B, 1).to(DEVICE)
    with torch.no_grad():
        det, mag, azm, reg, att, proj, gate = model(dummy_img, dummy_cosmic, prior_batch)
    print(f"  Azimuth Output Shape: {azm.shape}  (Expected: [B, 2])")
    print(f"  Gate Value: {gate:.4f}")
    print(f"  Azm norm: {azm.norm(dim=1).mean():.6f} (Expected: 1.0)")
    print("\n  [OK] V9 GATED ARCHITECTURE VERIFIED.")
    print("=" * 60)
