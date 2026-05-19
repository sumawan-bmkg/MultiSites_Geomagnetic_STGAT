#!/usr/bin/env python3
"""
V9_3_Bayesian_Model.py — V9.3 ARCHITECTURE: THE PHYSICS-INFORMED SIDECAR
=========================================================================
Modifications vs V9.2:
  1. REFREEZE THE BACKBONE: 100% frozen EfficientNet to prevent overfitting.
  2. PHYSICS BYPASS: Extract H, D, Z energy ratios directly from x_img.
  3. GATE BALANCING: Replace Prior-Dropout with Image-Dropout (p=0.5) to force prior usage.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR  = Path(__file__).parent
_ROOT_DIR  = _THIS_DIR.parent
_V8_DIR    = _ROOT_DIR / "ScalogramV3_V8_Repository" / "model"

sys.path.insert(0, str(_ROOT_DIR))
sys.path.insert(0, str(_V8_DIR))

from V3_Model_v8 import MultiTaskScalogramV3_v8


class BayesianAzimuthHead(nn.Module):
    """Azimuth head with Physics Sidecar and Image-Dropout Gate Balancing."""

    def __init__(self, img_dim: int = 512, prior_dim: int = 360):
        super().__init__()

        # [V9.3 TASK 3] Prior features expanded to 160-D to match Enhanced Image Features
        self.prior_proj = nn.Sequential(
            nn.Linear(prior_dim, 160),
            nn.LayerNorm(160),
            nn.ReLU(inplace=True)
        )
        # Removed prior_dropout

        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LayerNorm(128),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True)
        )

        # [V9.3 TASK 2] Physics Sidecar (3 ratios -> 32-D)
        self.physics_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True)
        )

        # [V9.3 TASK 3] Image-Dropout to balance the gate
        self.image_dropout = nn.Dropout(p=0.5)

        self.gate_net = nn.Sequential(
            nn.Linear(160 + 160, 160),
            nn.Sigmoid()
        )

        self.out_net = nn.Sequential(
            nn.Linear(160, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m == self.gate_net[0]:
                        nn.init.constant_(m.bias, 0.0) # Start balanced
                    else:
                        nn.init.constant_(m.bias, 0.0)

    def forward(self, img_feat: torch.Tensor, prior_vec: torch.Tensor, x_img: torch.Tensor):
        # [V9.3 TASK 2] Physics Bypass
        # x_img shape: (B, 3, 128, 1440). Channels: 0=H, 1=D, 2=Z
        mean_energy = torch.mean(torch.abs(x_img), dim=[2, 3]) # (B, 3)
        H = mean_energy[:, 0]
        D = mean_energy[:, 1]
        Z = mean_energy[:, 2]

        eps = 1e-6
        ratio_ZH = Z / (H + eps)
        ratio_ZD = Z / (D + eps)
        ratio_HD = H / (D + eps)

        physics_input = torch.stack([ratio_ZH, ratio_ZD, ratio_HD], dim=1) # (B, 3)
        physics_feat = self.physics_proj(physics_input) # (B, 32)

        # Image features
        i_feat_base = self.img_proj(img_feat) # (B, 128)

        # Combine backbone features with deterministic physics features
        enhanced_i_feat = torch.cat([i_feat_base, physics_feat], dim=-1) # (B, 160)

        # [V9.3 TASK 3] Image Dropout (Blindfold the image to rely on prior)
        enhanced_i_feat = self.image_dropout(enhanced_i_feat)

        # Prior features
        p_feat = self.prior_proj(prior_vec) # (B, 160)

        # Gate
        combined = torch.cat([enhanced_i_feat, p_feat], dim=-1) # (B, 320)
        gate = self.gate_net(combined) # (B, 160)

        if self.training:
            # Drop entire enhanced image feature randomly (10% of the time) like in V9.2
            img_mask = (torch.rand(enhanced_i_feat.size(0), 1, device=enhanced_i_feat.device) > 0.1).float()
            enhanced_i_feat = enhanced_i_feat * img_mask

        fused = (gate * enhanced_i_feat) + ((1.0 - gate) * p_feat)
        raw_out = self.out_net(fused)
        unit_vec = F.normalize(raw_out, p=2, dim=1)
        return unit_vec, gate.mean()


class MultiTaskScalogramV9_3_Bayesian(nn.Module):
    """ScalogramV9.3 — Physics-Informed Sidecar."""

    def __init__(self, prior_dim: int = 360, img_feat_dim: int = 512):
        super().__init__()
        self.prior_dim    = prior_dim
        self.img_feat_dim = img_feat_dim
        self._v8 = MultiTaskScalogramV3_v8(pretrained=False)
        self._freeze_v8()

        self.head_azimuth_bayesian = BayesianAzimuthHead(
            img_dim=img_feat_dim, prior_dim=prior_dim,
        )

    def _freeze_v8(self):
        """
        [V9.3 TASK 1]
        100% Frozen Backbone. No unfreezing.
        """
        frozen_params = 0
        unfrozen_params = 0

        for param in self._v8.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

        print(f"[OK] V9.3 Backbone: {frozen_params:,} completely frozen, {unfrozen_params:,} unfrozen.")

    def load_v8_checkpoint(self, ckpt_path: str) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            print(f"[WARN] Checkpoint not found: {path}")
            return
        ckpt = torch.load(str(path), map_location='cpu')
        sd = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
        sd_clean = {k.replace('module.', ''): v for k, v in sd.items()}
        self._v8.load_state_dict(sd_clean, strict=False)
        self._freeze_v8()

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
        out_azimuth, gate_val = self.head_azimuth_bayesian(v_fusion, prior_vec, x_img)
        return out_detection, out_magnitude, out_azimuth, reg_score, att_weights, proj_vec, gate_val


def build_v9_3_from_v8(ckpt_path, prior_dim=360, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskScalogramV9_3_Bayesian(prior_dim=prior_dim).to(device)
    model.load_v8_checkpoint(ckpt_path)
    return model


def azimuth_deg_from_sincos(pred: torch.Tensor) -> torch.Tensor:
    unit = F.normalize(pred, p=2, dim=1)
    deg  = torch.atan2(unit[:, 0], unit[:, 1]) * (180.0 / torch.pi)
    return deg % 360.0


if __name__ == "__main__":
    print("=" * 60)
    print("  V9.3 PHYSICS-INFORMED SIDECAR — SMOKE TEST")
    print("=" * 60)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskScalogramV9_3_Bayesian(prior_dim=360).to(DEVICE)
    B = 2
    dummy_img = torch.randn(B, 3, 128, 1440).to(DEVICE)
    dummy_cosmic = torch.randn(B, 2).to(DEVICE)
    prior_batch = (torch.ones(360) / 360.0).unsqueeze(0).repeat(B, 1).to(DEVICE)
    with torch.no_grad():
        det, mag, azm, reg, att, proj, gate = model(dummy_img, dummy_cosmic, prior_batch)
    print(f"  Azimuth: {azm.shape}  Gate: {gate:.4f}  Norm: {azm.norm(dim=1).mean():.6f}")
    print("  [OK] V9.3 ARCHITECTURE VERIFIED.")
    print("=" * 60)
