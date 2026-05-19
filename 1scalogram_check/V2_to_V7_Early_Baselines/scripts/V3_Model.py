import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scalogramv4.models.V4_GNN_Module import SpatialGNNModule

class MultiTaskScalogramV3(nn.Module):
    """
    ScalogramV3: MTL-CRNN with Spatial GNN Fusion and Cosmic Feature Injection.
    Inputs:
        - Image: (Batch, 3, 128, 1440) -> Continuous Wavelet Transform
        - Cosmic: (Batch, 2) -> [Kp Index, Dst Index]
    """
    def __init__(self, pretrained=True):
        super(MultiTaskScalogramV3, self).__init__()

        # 1. SPATIAL FEATURE EXTRACTOR (EfficientNet-B1)
        self.backbone = models.efficientnet_b1(weights='DEFAULT' if pretrained else None)
        self.features = self.backbone.features

        # 2. TEMPORAL SEQUENCE MODEL (BiGRU)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        self.gru = nn.GRU(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 3. SPATIAL GNN BRIDGE (Ported from V4)
        # V3 now treats the BiGRU output (512) as the node feature for GNN.
        # We assume a fixed 8-station layout for regional consensus.
        self.gnn = SpatialGNNModule(in_features=512, hidden=256, out_features=512, n_heads=4)

        # 4. AUXILIARY COSMIC HEAD -> ATTENTION GENERATOR
        self.cosmic_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 512), # Match GNN output dim
            nn.Sigmoid()
        )

        # 5. TASK-SPECIFIC HEADS
        self.fusion_dim = 512

        self.head_detection = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

        self.head_magnitude = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

        self.head_azimuth = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.apply(self._init_weights)
        nn.init.constant_(self.cosmic_mlp[-2].bias, 3.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x_img, x_cosmic, x_mask=None, T=1.0):
        # 1. Extract Spatial Features
        x = self.features(x_img)

        # 2. Sequence Modeling
        x = self.adaptive_pool(x)
        x = x.squeeze(2).permute(0, 2, 1)

        if x_mask is not None:
            target_seq_len = x.size(1)
            x_mask_pooled = F.adaptive_avg_pool1d(x_mask.unsqueeze(1), target_seq_len)
            x_mask_pooled = (x_mask_pooled > 0.5).float()
            x = x * x_mask_pooled.permute(0, 2, 1)

        self.gru.flatten_parameters()
        gru_out, _ = self.gru(x)  # (B, 45, 512)

        # Global Temporal Pooling -> Node Feature
        v_img = torch.mean(gru_out, dim=1) # (B, 512)

        # 3. GNN SPATIAL FUSION
        # In V3, we simulate N=8 nodes by expanding v_img or using a multi-station batch.
        # Since V3 typically processes one station image at a time,
        # we treat the current sample as the 'active' node in a virtual 8-node graph.
        B = v_img.shape[0]
        # Expand to (B, 8, 512) - in a real multi-station setup, these would be distinct station features.
        station_features = v_img.unsqueeze(1).repeat(1, 8, 1)
        station_probs = torch.ones(B, 8, device=v_img.device) * 0.5 # Default confidence

        # GNN output: consensus_feat (B, 128), regional_score (B, 1), adaptive_th (B, 1), att_weights (B, 8)
        consensus_feat, reg_score, th, att_weights = self.gnn(station_features, station_probs)

        # 4. Cosmic Gating
        cosmic_attention = self.cosmic_mlp(x_cosmic)
        v_fusion = consensus_feat * cosmic_attention

        # 5. Predict
        out_detection = self.head_detection(v_fusion)
        out_magnitude = self.head_magnitude(v_fusion)
        out_azimuth = self.head_azimuth(v_fusion)

        # Temperature Scaling for Detection Probabilities
        if T != 1.0:
            out_detection = out_detection / T

        return out_detection, out_magnitude, out_azimuth, reg_score, att_weights

if __name__ == "__main__":
    model = MultiTaskScalogramV3(pretrained=False)
    dummy_img = torch.randn(2, 3, 128, 1440)
    dummy_cosmic = torch.randn(2, 2)

    det, mag, azm, reg, att = model(dummy_img, dummy_cosmic)
    print("ScalogramV3 GNN-Fused Forward Pass Success!")
    print(f"Detection shape: {det.shape}")
    print(f"Magnitude shape: {mag.shape}")
    print(f"Azimuth shape:   {azm.shape}")
    print(f"Regional Score:  {reg.shape}")
    print(f"Attention:       {att.shape}")
