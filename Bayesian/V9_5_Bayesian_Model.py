#!/usr/bin/env python3
"""
V9_5_Bayesian_Model.py — V9.5 REVISION: CONDITIONAL STATION-MODULATED PHYSICS (CSMP)
===================================================================================
1. STATION EMBEDDING: Upgraded to 32-D.
2. CONDITIONAL ATTENTION: Station identity modulates temporal focus.
3. STATION-AWARE GATING: Gate net sees station context.
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
    rad = torch.atan2(sin_cos_tensor[:, 0], sin_cos_tensor[:, 1])
    deg = torch.rad2deg(rad)
    return (deg + 360) % 360

class ConditionalTemporalAttention(nn.Module):
    """
    [V9.5 TASK 2]
    Station-Aware Temporal Attention. 
    Uses station context to decide where to focus in time.
    """
    def __init__(self, stn_dim=32):
        super().__init__()
        # Input: time_series (3) + stn_feat (32) = 35 channels
        self.attention_net = nn.Sequential(
            nn.Conv1d(3 + stn_dim, 16, kernel_size=15, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 8, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.physics_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_img: torch.Tensor, stn_feat: torch.Tensor):
        # x_img: [B, 3, 128, 1440], stn_feat: [B, 32]
        
        # 1. Frequency Pooling -> [B, 3, 1440]
        time_series = torch.mean(torch.abs(x_img), dim=2) 
        
        # 2. Expand Station Feature to Timeline -> [B, 32, 1440]
        B, C, T = time_series.shape
        stn_feat_expanded = stn_feat.unsqueeze(-1).expand(B, -1, T)
        
        # 3. Concat Physics + Station Context -> [B, 35, 1440]
        combined_input = torch.cat([time_series, stn_feat_expanded], dim=1)
        
        # 4. Compute Conditional Attention Weights
        attn_weights = self.attention_net(combined_input) # [B, 1, 1440]
        
        # 5. Weighted Aggregation -> [B, 3]
        weighted_energy = torch.sum(time_series * attn_weights, dim=-1)
        
        eps = 1e-6
        H, D, Z = weighted_energy[:, 0], weighted_energy[:, 1], weighted_energy[:, 2]
        ratios = torch.stack([Z/(H+eps), Z/(D+eps), H/(D+eps)], dim=1)
        
        return self.physics_proj(ratios)

class BayesianAzimuthHeadV95(nn.Module):
    def __init__(self, img_dim=512, prior_dim=360, num_stations=23):
        super().__init__()
        
        # 1. Station Embedding (32-D)
        self.station_embedding = nn.Embedding(num_stations, 32)
        
        # 2. Conditional Physics (32-D)
        self.physics_encoder = ConditionalTemporalAttention(stn_dim=32)
        
        # 3. Prior Encoding (160-D)
        self.prior_proj = nn.Sequential(
            nn.Linear(prior_dim, 160),
            nn.LayerNorm(160),
            nn.ReLU(inplace=True)
        )
        
        # 4. Image Encoding (128-D)
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True)
        )
        
        self.image_dropout = nn.Dropout(p=0.3)
        
        # Fusion: Img(128) + Phys(32) + Stn(32) + Prior(160) = 352
        total_dim = 128 + 32 + 32 + 160
        
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
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, img_feat, prior_vec, x_img, station_id):
        # A. Station context
        stn_feat = self.station_embedding(station_id) # [B, 32]
        
        # B. Conditional Physics
        phys_feat = self.physics_encoder(x_img, stn_feat) # [B, 32]
        
        # C. Visual & Prior features
        i_feat = self.img_proj(img_feat) # [B, 128]
        p_feat = self.prior_proj(prior_vec) # [B, 160]
        
        # D. Fusion
        all_features = torch.cat([i_feat, phys_feat, stn_feat, p_feat], dim=-1) # [B, 352]
        all_features = self.image_dropout(all_features)
        
        gate = self.gate_net(all_features)
        raw_out = self.out_net(all_features)
        unit_vec = F.normalize(raw_out, p=2, dim=1)
        
        return unit_vec, gate.mean()

class MultiTaskScalogramV9_5_Bayesian(nn.Module):
    def __init__(self, prior_dim=360, img_feat_dim=512, num_stations=23):
        super().__init__()
        self._v8 = MultiTaskScalogramV3_v8(pretrained=False)
        for param in self._v8.parameters():
            param.requires_grad = False
            
        self.head_azimuth_bayesian = BayesianAzimuthHeadV95(
            img_dim=img_feat_dim, prior_dim=prior_dim, num_stations=num_stations
        )

    def load_v8_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
        sd_clean = {k.replace('module.', ''): v for k, v in sd.items()}
        self._v8.load_state_dict(sd_clean, strict=False)

    def forward(self, x_img, x_cosmic, prior_vec, station_id, T=1.0):
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
        
        out_detection = v8.head_detection(v_fusion) / T
        out_magnitude = v8.head_magnitude(v_fusion)
        
        return out_detection, out_magnitude, out_azimuth, gate_val

if __name__ == "__main__":
    print("V9.5 SMOKE TEST...")
    model = MultiTaskScalogramV9_5_Bayesian()
    B = 2
    dummy_img = torch.randn(B, 3, 128, 1440)
    dummy_cosmic = torch.randn(B, 2)
    dummy_prior = torch.randn(B, 360)
    dummy_stn = torch.randint(0, 23, (B,))
    det, mag, azm, gate = model(dummy_img, dummy_cosmic, dummy_prior, dummy_stn)
    print(f"Azm Shape: {azm.shape}, Gate: {gate.item():.4f}")
    print("[OK] V9.5 Architecture Verified.")
