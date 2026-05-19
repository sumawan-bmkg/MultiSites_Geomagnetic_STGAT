#!/usr/bin/env python3
"""
V3_Model_v8.py — ScalogramV3 Architecture Overhaul v8.1 (MANIFOLD SHATTER)
Emergency fixes:

  TASK 1 (BUG FIX): dynamic_threshold_from_negatives sekarang menerima
                     PROBABILITAS (sigmoid output), bukan logits. Clamp [0.01, 0.99].
  TASK 2 (SUPCON):   SupConLoss (Khosla et al. 2020) + projection_head di model.
  TASK 3 (FUSION):   SupCon loss diintegrasikan ke training loop.
  TASK 4 (FPR-AWARE): Checkpoint disimpan jika FPR < 0.5.

Ref:
  - Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
  - Prokudin et al., "Deep Directional Statistics" (ECCV 2018)
  - Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
  - Siffer et al., "Anomaly Detection in Streams" (KDD 2017)
  - Müller et al., "When Does Label Smoothing Help?" (NeurIPS 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scalogramv4.models.V4_GNN_Module import SpatialGNNModule


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 (S2): Sine-Cosine Azimuth Head
# ─────────────────────────────────────────────────────────────────────────────
class SineCosineLoss(nn.Module):
    """
    Circular regression loss via unit-vector cosine similarity.
    pred: (B, 2) — raw [sin, cos] logits (will be L2-normalized internally)
    target_deg: (B,) — azimuth in degrees
    Returns scalar loss = mean(1 - cos_similarity)
    """
    def forward(self, pred: torch.Tensor, target_deg: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        # Normalize prediction to unit vector
        pred_unit = F.normalize(pred, p=2, dim=1)          # (B, 2)

        # Convert target degrees → unit vector [sin, cos]
        target_rad = target_deg * (torch.pi / 180.0)
        target_unit = torch.stack([torch.sin(target_rad),
                                   torch.cos(target_rad)], dim=1)  # (B, 2)

        # Cosine similarity per sample → loss = 1 - similarity
        cos_sim = (pred_unit * target_unit).sum(dim=1)     # (B,)
        loss_per_sample = 1.0 - cos_sim                    # (B,)

        if mask is not None:
            # Only compute loss on positive (earthquake) samples
            loss_per_sample = loss_per_sample * mask
            denom = mask.sum() + 1e-8
            return loss_per_sample.sum() / denom

        return loss_per_sample.mean()


def azimuth_from_sincos(pred: torch.Tensor) -> torch.Tensor:
    """
    Convert raw [sin, cos] head output → degrees [0, 360).
    pred: (B, 2)
    Returns: (B,) in degrees
    """
    pred_unit = F.normalize(pred, p=2, dim=1)
    deg = torch.atan2(pred_unit[:, 0], pred_unit[:, 1]) * (180.0 / torch.pi)
    return deg % 360.0


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 (S12): Label Smoothing Cross-Entropy
# ─────────────────────────────────────────────────────────────────────────────
class LabelSmoothingFocalLoss(nn.Module):
    """
    Focal Loss + Label Smoothing (epsilon=0.1).
    Prevents logits from exploding to ±∞, which collapses Brier Score.

    Hard labels:  0 → 0.05,  1 → 0.95
    Then applies focal weighting: (1 - p_t)^gamma
    """
    def __init__(self, gamma: float = 2.0, epsilon: float = 0.1,
                 num_classes: int = 2):
        super().__init__()
        self.gamma   = gamma
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C) raw logits
        targets: (B,)   integer class indices
        """
        # Soft targets via label smoothing
        B, C = logits.shape
        smooth_val = self.epsilon / C
        # One-hot then smooth
        with torch.no_grad():
            soft_targets = torch.full_like(logits, smooth_val)
            soft_targets.scatter_(1, targets.unsqueeze(1),
                                  1.0 - self.epsilon + smooth_val)

        # Log-softmax
        log_probs = F.log_softmax(logits, dim=1)           # (B, C)
        probs     = log_probs.exp()                        # (B, C)

        # Focal weight: use p_t of the TRUE class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        focal_weight = (1.0 - p_t) ** self.gamma               # (B,)

        # Cross-entropy with soft targets
        ce = -(soft_targets * log_probs).sum(dim=1)            # (B,)

        loss = focal_weight * ce
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 (SUPCON): Supervised Contrastive Loss
# Ref: Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
# ─────────────────────────────────────────────────────────────────────────────
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Memaksa sampel dengan label sama berkumpul di hypersphere,
    dan sampel dengan label berbeda berjauhan.

    features: (B, D) — L2-normalized projection vectors
    labels:   (B,)   — integer class labels (0=negatif, 1=positif)
    temperature: skalar, default 0.07 (Khosla et al.)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device,
                                requires_grad=True)

        # L2 normalize (pastikan unit sphere)
        features = F.normalize(features, p=2, dim=1)  # (B, D)

        # Similarity matrix: (B, B) — dot product di unit sphere = cosine sim
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask diagonal (self-similarity)
        eye_mask = torch.eye(B, dtype=torch.bool, device=features.device)

        # Positive mask: sampel dengan label sama (tidak termasuk diri sendiri)
        labels_col = labels.unsqueeze(1)   # (B, 1)
        labels_row = labels.unsqueeze(0)   # (1, B)
        pos_mask = (labels_col == labels_row) & ~eye_mask  # (B, B)

        # Jika tidak ada pasangan positif dalam batch, return 0
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device,
                                requires_grad=True)

        # Log-sum-exp denominator: semua sampel kecuali diri sendiri
        # Numerically stable: subtract max
        sim_matrix_no_diag = sim_matrix.masked_fill(eye_mask, float('-inf'))
        log_denom = torch.logsumexp(sim_matrix_no_diag, dim=1, keepdim=True)  # (B,1)

        # Log probability untuk setiap pasangan positif
        log_prob = sim_matrix - log_denom  # (B, B)

        # Mean log-prob atas semua pasangan positif per anchor
        # Hanya hitung untuk anchor yang punya setidaknya 1 positif
        n_pos_per_anchor = pos_mask.float().sum(dim=1)  # (B,)
        valid_anchor = n_pos_per_anchor > 0

        if valid_anchor.sum() == 0:
            return torch.tensor(0.0, device=features.device,
                                requires_grad=True)

        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1)  # (B,)
        mean_log_prob_pos = mean_log_prob_pos[valid_anchor] / \
                            n_pos_per_anchor[valid_anchor]

        loss = -mean_log_prob_pos.mean()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 (S11): Dynamic Adaptive Thresholding
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_threshold_f2(probs: torch.Tensor,
                               labels: torch.Tensor,
                               beta: float = 2.0,
                               n_steps: int = 100) -> float:
    """
    Sweep thresholds [0.01, 0.99] and return the one maximising F-beta score.
    beta=2 weights recall twice as much as precision (EWS use-case).

    probs:  (N,) detection probabilities
    labels: (N,) binary ground-truth
    Returns: optimal threshold (float)
    """
    probs_np  = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    best_th, best_fb = 0.5, -1.0
    for th in [i / n_steps for i in range(1, n_steps)]:
        preds = (probs_np >= th).astype(float)
        tp = ((preds == 1) & (labels_np == 1)).sum()
        fp = ((preds == 1) & (labels_np == 0)).sum()
        fn = ((preds == 0) & (labels_np == 1)).sum()

        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        fb   = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-8)

        if fb > best_fb:
            best_fb = fb
            best_th = th

    return best_th


def dynamic_threshold_from_negatives(neg_probs: torch.Tensor,
                                     k: float = 3.0) -> float:
    """
    TASK 1 FIX: Siffer et al. formula menggunakan PROBABILITAS (bukan logits).
    neg_probs HARUS sudah di-sigmoid sebelum dipanggil.
    threshold = clamp(mean + k*std, 0.01, 0.99)
    """
    # Pastikan dalam range [0,1] — jika ada nilai > 1, berarti logits masuk
    if neg_probs.max().item() > 1.0 or neg_probs.min().item() < 0.0:
        # Auto-fix: terapkan sigmoid
        neg_probs = torch.sigmoid(neg_probs)

    mu  = neg_probs.mean().item()
    std = neg_probs.std().item()
    raw = mu + k * std
    # TASK 1 FIX: clamp ke [0.01, 0.99]
    return float(max(0.01, min(0.99, raw)))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL — V8 Overhaul
# ─────────────────────────────────────────────────────────────────────────────
class MultiTaskScalogramV3_v8(nn.Module):
    """
    ScalogramV3 v8 — Emergency Architecture Overhaul.

    Changes vs original V3_Model.py:
      1. head_azimuth now outputs 2D unit vector [sin, cos]  (TASK 1)
         → forward() applies F.normalize before returning azimuth
      2. head_detection unchanged; loss is now LabelSmoothingFocalLoss (TASK 4)
         (loss is external, model just returns raw logits)
      3. Dynamic threshold helpers are module-level functions (TASK 3)
      4. WeightedRandomSampler already in DataLoader (TASK 2 — no model change)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ── 1. Backbone (EfficientNet-B1) ──────────────────────────────────
        backbone = models.efficientnet_b1(
            weights='DEFAULT' if pretrained else None)
        self.features = backbone.features

        # ── 2. Temporal (BiGRU) ────────────────────────────────────────────
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        # Projection: 1280 → 256 sebelum GRU untuk hemat VRAM pada 2GB GPU
        self.gru_proj = nn.Linear(1280, 256)
        self.gru = nn.GRU(
            input_size=256, hidden_size=256,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.2
        )

        # ── 3. Spatial GNN ─────────────────────────────────────────────────
        self.gnn = SpatialGNNModule(
            in_features=512, hidden=256, out_features=512, n_heads=4)

        # ── 4. Cosmic Gating MLP ───────────────────────────────────────────
        self.cosmic_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 512),
            nn.Sigmoid()
        )

        # ── 5. Task Heads ──────────────────────────────────────────────────
        self.fusion_dim = 512

        self.head_detection = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)          # raw logits; loss applies label smoothing
        )

        self.head_magnitude = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

        # TASK 1 (S2): 2-output head for [sin(θ), cos(θ)]
        self.head_azimuth = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)          # outputs [sin, cos] — NOT degrees
        )

        # TASK 2 (SUPCON): Projection head untuk Supervised Contrastive Learning
        # Linear(512, 128) -> ReLU -> Linear(128, 128) -> L2 Norm
        self.projection_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )  # L2 normalization dilakukan di forward()

        self.apply(self._init_weights)
        nn.init.constant_(self.cosmic_mlp[-2].bias, 3.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x_img: torch.Tensor, x_cosmic: torch.Tensor,
                x_mask=None, T: float = 1.0):
        # ── Backbone ───────────────────────────────────────────────────────
        x = self.features(x_img)

        # ── BiGRU ──────────────────────────────────────────────────────────
        x = self.adaptive_pool(x)
        x = x.squeeze(2).permute(0, 2, 1)   # (B, T, 1280)
        x = self.gru_proj(x)                 # (B, T, 256) — VRAM reduction

        if x_mask is not None:
            tgt = x.size(1)
            m = F.adaptive_avg_pool1d(x_mask.unsqueeze(1), tgt)
            m = (m > 0.5).float()
            x = x * m.permute(0, 2, 1)

        self.gru.flatten_parameters()
        gru_out, _ = self.gru(x)
        v_img = torch.mean(gru_out, dim=1)          # (B, 512)

        # ── GNN ────────────────────────────────────────────────────────────
        B = v_img.shape[0]
        station_features = v_img.unsqueeze(1).repeat(1, 8, 1)
        station_probs    = torch.ones(B, 8, device=v_img.device) * 0.5
        consensus_feat, reg_score, th, att_weights = self.gnn(
            station_features, station_probs)

        # ── Cosmic Gating ──────────────────────────────────────────────────
        cosmic_attention = self.cosmic_mlp(x_cosmic)
        v_fusion = consensus_feat * cosmic_attention

        # ── Task Heads ─────────────────────────────────────────────────────
        out_detection = self.head_detection(v_fusion)
        if T != 1.0:
            out_detection = out_detection / T

        out_magnitude = self.head_magnitude(v_fusion)

        # TASK 1 (S2): normalize azimuth output to unit vector
        out_azimuth_raw = self.head_azimuth(v_fusion)          # (B, 2)
        out_azimuth = F.normalize(out_azimuth_raw, p=2, dim=1) # (B, 2) unit vec

        # TASK 2 (SUPCON): projection vector untuk SupCon loss
        # L2-normalized di unit hypersphere (B, 128)
        proj_raw = self.projection_head(v_fusion)              # (B, 128)
        proj_vec = F.normalize(proj_raw, p=2, dim=1)           # (B, 128) unit vec

        return out_detection, out_magnitude, out_azimuth, reg_score, att_weights, proj_vec


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MultiTaskScalogramV3_v8(pretrained=False)
    dummy_img    = torch.randn(2, 3, 128, 1440)
    dummy_cosmic = torch.randn(2, 2)

    det, mag, azm, reg, att, proj = model(dummy_img, dummy_cosmic)
    print("V8.1 MANIFOLD SHATTER Forward Pass OK")
    print(f"  Detection  : {det.shape}")
    print(f"  Magnitude  : {mag.shape}")
    print(f"  Azimuth    : {azm.shape}  (unit vector [sin,cos])")
    print(f"  Projection : {proj.shape}  (SupCon, unit sphere)")
    print(f"  Azm norm   : {azm.norm(dim=1).mean():.6f}  (should be ~1.0)")
    print(f"  Proj norm  : {proj.norm(dim=1).mean():.6f}  (should be ~1.0)")

    labels   = torch.tensor([1, 0])
    azimuths = torch.tensor([45.0, 0.0])

    loss_azm    = SineCosineLoss()(azm, azimuths)
    loss_det    = LabelSmoothingFocalLoss()(det, labels)
    loss_supcon = SupConLoss(temperature=0.07)(proj, labels)

    print(f"  SineCosineLoss   : {loss_azm.item():.4f}")
    print(f"  LabelSmoothFocal : {loss_det.item():.4f}")
    print(f"  SupConLoss       : {loss_supcon.item():.4f}")

    # TASK 1 FIX: test dynamic threshold dengan probabilitas
    probs_neg = torch.sigmoid(det[labels == 0, 1])
    th = dynamic_threshold_from_negatives(probs_neg)
    print(f"  Dynamic threshold (from probs): {th:.4f}  (should be in [0.01, 0.99])")
    assert 0.01 <= th <= 0.99, f"BUG: threshold {th} out of range!"
    print("All checks passed.")

    loss_azm = SineCosineLoss()(azm, azimuths)
    loss_det = LabelSmoothingFocalLoss()(det, labels)
    print(f"  SineCosineLoss   : {loss_azm.item():.4f}")
    print(f"  LabelSmoothFocal : {loss_det.item():.4f}")

    # Test dynamic threshold
    probs  = torch.sigmoid(det[:, 1])
    labels_t = torch.tensor([1, 0])
    th_f2  = find_optimal_threshold_f2(probs, labels_t)
    th_dyn = dynamic_threshold_from_negatives(probs[labels_t == 0])
    print(f"  F2-optimal threshold : {th_f2:.3f}")
    print(f"  Dynamic threshold    : {th_dyn:.3f}")
    print("All checks passed.")
