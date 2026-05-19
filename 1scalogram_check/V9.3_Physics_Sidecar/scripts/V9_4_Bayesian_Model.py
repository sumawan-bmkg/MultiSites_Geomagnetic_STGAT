#!/usr/bin/env python3
"""
V9_4_Bayesian_Model.py — V9.4 ULTIMATE: STATION EMBEDDING & TEMPORAL ATTENTION
=============================================================================
1. BACKBONE: 100% Frozen.
2. STATION EMBEDDING: Correction for Local Crustal Anomalies.
3. TEMPORAL ATTENTION: Focusing on specific minutes of the 1440m window.
4. FINAL FUSION: Image(128) + Physics(32) + Embedding(16) + Prior(160) = 336-D.
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

def azimuth_deg_from_sincos(sin_cos_tensor: torch.Tensor) -> torch.Tensor:
    """Converts [sin, cos] output to degrees [0, 360)."""
    rad = torch.atan2(sin_cos_tensor[:, 0], sin_cos_tensor[:, 1])
    deg = torch.rad2deg(rad)
    return (deg + 360) % 360

class TemporalPhysicsEncoder(nn.Module):
    """
    [V9.4 TASK 2]
    Extracts physics features with Temporal Attention focusing.
    """
    def __init__(self):
        super().__init__()
        # Attention on the 1440 minutes timeline
        self.attention_net = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.physics_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_img: torch.Tensor):
        # x_img: [B, 3, 128, 1440]
        # 1. Frequency Pooling (Reduce 128 freq bins to 1)
        # We use mean absolute value to represent energy
        time_series = torch.mean(torch.abs(x_img), dim=2) # [B, 3, 1440]
        
        # 2. Compute Attention Weights
        attn_weights = self.attention_net(time_series) # [B, 1, 1440]
        
        # 3. Apply Attention & Global Sum
        # weighted_energy: [B, 3]
        weighted_energy = torch.sum(time_series * attn_weights, dim=-1) # [B, 3]
        
        H = weighted_energy[:, 0]
        D = weighted_energy[:, 1]
        Z = weighted_energy[:, 2]
        
        eps = 1e-6
        ratio_ZH = Z / (H + eps)
        ratio_ZD = Z / (D + eps)
        ratio_HD = H / (D + eps)
        
        physics_input = torch.stack([ratio_ZH, ratio_ZD, ratio_HD], dim=1) # [B, 3]
        physics_feat = self.physics_proj(physics_input) # [B, 32]
        
        return physics_feat

class BayesianAzimuthHead(nn.Module):
    """Azimuth head with Station Embedding and Temporal Attention."""

    def __init__(self, img_dim: int = 512, prior_dim: int = 360, num_stations: int = 23):
        super().__init__()

        # 1. Prior Encoding (160-D)
        self.prior_proj = nn.Sequential(
            nn.Linear(prior_dim, 160),
            nn.LayerNorm(160),
            nn.ReLU(inplace=True)
        )

        # 2. Image Encoding (128-D)
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LayerNorm(128),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True)
        )

        # 3. Station Embedding (16-D) [TASK 1]
        self.station_embedding = nn.Embedding(num_stations, 16)

        # 4. Temporal Physics (32-D) [TASK 2]
        self.physics_encoder = TemporalPhysicsEncoder()

        # 5. Image Dropout for Balancing
        self.image_dropout = nn.Dropout(p=0.3)

        # Fusion Input Dim: 128 (Img) + 32 (Phys) + 16 (Stn) + 160 (Prior) = 336
        total_dim = 128 + 32 + 16 + 160
        
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, 160),
            nn.Sigmoid()
        )

        self.out_net = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, img_feat: torch.Tensor, prior_vec: torch.Tensor, x_img: torch.Tensor, station_id: torch.Tensor):
        # 1. Station Context
        stn_feat = self.station_embedding(station_id) # [B, 16]
        
        # 2. Physics Context (Temporal Attention)
        phys_feat = self.physics_encoder(x_img) # [B, 32]
        
        # 3. Image Context (Backbone)
        i_feat_base = self.img_proj(img_feat) # [B, 128]
        
        # 4. Prior Context
        p_feat = self.prior_proj(prior_vec) # [B, 160]
        
        # Combine Image-related features
        enhanced_i_feat_full = torch.cat([i_feat_base, phys_feat, stn_feat], dim=-1) # [B, 128+32+16] = [B, 176]
        enhanced_i_feat_full = self.image_dropout(enhanced_i_feat_full)
        
        # Final Fusion with Prior
        # Note: gate is 160-D, so we might need to adjust logic or concat
        # In V9.4 we combine ALL for the gate calculation
        all_features = torch.cat([i_feat_base, phys_feat, stn_feat, p_feat], dim=-1) # [B, 336]
        
        # Logic: gate decides how much to trust the PRIOR vs the OTHERS
        # But for simplicity in this V9.4, we use a single out_net on combined features
        # and keep the gate as a diagnostic value.
        gate = self.gate_net(all_features) 
        
        # Traditional Gated Fusion for 160-D features?
        # Let's use the gated approach for the 160-D prior vs the 176-D enhanced image features
        # Actually, let's keep it simple as requested: "Final Fusion"
        raw_out = self.out_net(all_features)
        unit_vec = F.normalize(raw_out, p=2, dim=1)
        
        return unit_vec, gate.mean()

class MultiTaskScalogramV9_4_Bayesian(nn.Module):
    def __init__(self, prior_dim: int = 360, img_feat_dim: int = 512, num_stations: int = 23):
        super().__init__()
        self._v8 = MultiTaskScalogramV3_v8(pretrained=False)
        for param in self._v8.parameters():
            param.requires_grad = False
            
        self.head_azimuth_bayesian = BayesianAzimuthHead(
            img_dim=img_feat_dim, prior_dim=prior_dim, num_stations=num_stations
        )

    def load_v8_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
        sd_clean = {k.replace('module.', ''): v for k, v in sd.items()}
        self._v8.load_state_dict(sd_clean, strict=False)

    def forward(self, x_img, x_cosmic, prior_vec, station_id, x_mask=None, T=1.0):
        v8 = self._v8
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
        
        out_azimuth, gate_val = self.head_azimuth_bayesian(v_fusion, prior_vec, x_img, station_id)
        
        # Magnitude & Detection from Frozen Backbone
        out_detection = v8.head_detection(v_fusion) / T
        out_magnitude = v8.head_magnitude(v_fusion)
        
        return out_detection, out_magnitude, out_azimuth, gate_val

if __name__ == "__main__":
    print("V9.4 SMOKE TEST...")
    model = MultiTaskScalogramV9_4_Bayesian()
    B = 2
    dummy_img = torch.randn(B, 3, 128, 1440)
    dummy_cosmic = torch.randn(B, 2)
    dummy_prior = torch.randn(B, 360)
    dummy_stn = torch.randint(0, 23, (B,))
    det, mag, azm, gate = model(dummy_img, dummy_cosmic, dummy_prior, dummy_stn)
    print(f"Azm Shape: {azm.shape}, Gate: {gate.item():.4f}")
    print("[OK] V9.4 Architecture Verified.")
