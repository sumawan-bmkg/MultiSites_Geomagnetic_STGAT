# Bayesian Spatial Prior Phase — V9 ScalogramV3

Folder ini berisi implementasi **Physics-Informed Bayesian Prior Integration**
untuk mengatasi *Mode Collapse* pada Azimuth Head akibat trade-off dengan SupCon Loss.

## Motivasi

Model ScalogramV3 V8 mengalami *mode collapse* pada prediksi azimuth karena:
- SupCon Loss mendominasi gradien dan menekan sinyal dari azimuth head
- Head azimuth tidak mendapat prior spasial tentang di mana gempa historis terjadi

Solusi: **Two-Stage Fine-Tuning** dengan prior Bayesian berbasis seismisitas historis.

---

## Struktur Folder

```
Bayesian/
├── build_spatial_prior.py          # MODULE 1: Generator Prior Spasial
├── V9_Bayesian_Model.py            # MODULE 2: Arsitektur V9 (Grafted Head)
├── train_azimuth_bayesian.py       # MODULE 3: Training Script
├── spatial_prior_1d.pt             # [OUTPUT] Prior tensor [360] (setelah build)
├── prior_distribution_polar.png    # [OUTPUT] Visualisasi polar prior
├── v9_bayesian_azimuth_best.pth    # [OUTPUT] Checkpoint terbaik (setelah train)
├── training_history.csv            # [OUTPUT] Log per-epoch
├── bayesian_training_curves.png    # [OUTPUT] Kurva training
├── bayesian_training.log           # [OUTPUT] Log teks lengkap
└── README.md                       # Dokumen ini
```

---

## Cara Penggunaan

### Step 1 — Build Spatial Prior

```bash
cd d:\multi\scalogramv3\Bayesian

python build_spatial_prior.py \
    --catalog  d:\multi\scalogramv3\2026.csv \
    --stn-lat  0.8126 \
    --stn-lon  127.3670 \
    --bandwidth 15.0 \
    --output-dir .
```

Output:
- `spatial_prior_1d.pt` — tensor [360] probability distribution azimuth
- `prior_distribution_polar.png` — polar plot distribusi

### Step 2 — (Opsional) Verifikasi Arsitektur V9

```bash
python V9_Bayesian_Model.py
```

Output yang diharapkan:
```
V9 BAYESIAN MODEL — SMOKE TEST
  Detection  : torch.Size([2, 2])   ✓
  Azimuth    : torch.Size([2, 2])   ✓  (unit vector)
  ✅ SEMUA CHECK LULUS  —  V9 Bayesian Model SIAP.
```

### Step 3 — Fine-Tuning

```bash
python train_azimuth_bayesian.py \
    --h5-path   d:\multi\scalogramv3\scalogram_v3_cosmic_final.h5 \
    --ckpt-v8   d:\multi\scalogramv3\checkpoints\v3_best_fusion_model.pth \
    --prior-pt  d:\multi\scalogramv3\Bayesian\spatial_prior_1d.pt \
    --output-dir d:\multi\scalogramv3\Bayesian \
    --epochs    15 \
    --lr        1e-4 \
    --batch-size 16 \
    --loss      sincos \
    --mask-positives
```

Untuk menggunakan VonMises Loss:
```bash
    --loss vonmises --vonmises-kappa 2.0
```

---

## Desain Arsitektur

### Frozen Layers (requires_grad = False)
| Layer               | Alasan                                       |
|---------------------|----------------------------------------------|
| EfficientNet-B1     | Pre-trained, tidak perlu diubah              |
| BiGRU (+ proj)      | Representasi temporal sudah optimal          |
| Spatial GNN         | Konsensus spasial sudah stabil               |
| Cosmic Gating MLP   | Sudah dikalibrasi dengan Kp/Dst              |
| Detection Head      | FPR 0.125 — tidak boleh terganggu            |
| Magnitude Head      | Sudah stabil                                 |
| SupCon Head         | Kontributor mode collapse — dibekukan        |

### Trainable Layer (Fine-Tuned)
| Layer                   | Dimensi Input        | Output |
|-------------------------|----------------------|--------|
| `BayesianAzimuthHead`   | 512 + 360 = **872**  | 2 (sin/cos) |

```
v_fusion  (B, 512)  ─┐
                      ├─ Concat → (B, 872) → Linear(512) → ReLU → Dropout(0.3)
prior_vec (B, 360)  ─┘              → Linear(128) → ReLU → Linear(2) → L2Norm
                                                              ↓
                                                     [sin θ, cos θ]  (unit vector)
```

---

## Loss Functions

### SineCosineLoss (default)
```
L = 1 - cos_similarity(pred_unit, target_unit)
```
Proper circular regression — tidak ada singularity di 0°/360°.

### VonMisesLoss (alternatif)
```
L = κ × (1 - cos(θ_pred - θ_true))
```
NLL dari distribusi Von Mises — gradien lebih tajam, cocok untuk distribusi unimodal.

---

## Referensi

- Khosla et al., *Supervised Contrastive Learning* (NeurIPS 2020)
- Prokudin et al., *Deep Directional Statistics* (ECCV 2018)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (ICLR 2019)
- Box & Tiao, *Bayesian Inference in Statistical Analysis* (1973)
