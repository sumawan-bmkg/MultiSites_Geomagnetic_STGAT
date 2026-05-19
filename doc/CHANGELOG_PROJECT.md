# Project Development Changelog

Dokumentasi lengkap setiap perubahan dan milestone dalam proyek ANTIGRAVITY.

## 2026-05-02: Project Initialization

### Struktur Proyek Dibuat

#### Direktori Utama
- ✅ `data/` - Storage untuk raw, processed, train, val, test data
- ✅ `models/` - Model checkpoints dan logs
- ✅ `scripts/` - Python scripts untuk data processing
- ✅ `config/` - File konfigurasi YAML
- ✅ `plots/` - Visualisasi dan grafik
- ✅ `references/` - Paper dan referensi penelitian
- ✅ `doc/` - Dokumentasi lengkap
- ✅ `notebooks/` - Jupyter notebooks untuk eksplorasi

### Dokumentasi Lengkap

#### File Dokumentasi Dibuat
1. **doc/00_project_overview.md**
   - Tujuan proyek dan arsitektur high-level
   - Dataset description (24 stasiun, 3 klaster tektonik)
   - Multi-task learning objectives
   - Strict chronological split strategy
   - Inovasi utama: Multi-station processing, tectonic-aware adjacency, physics-informed loss

2. **doc/01_data_cleaning.md**
   - Prosedur cleaning untuk lokasi_stasiun.csv
   - Prosedur cleaning untuk earthquake_catalog
   - Validasi koordinat dan data quality checks
   - Removal of training artifacts (numeric station codes)
   - Expected output dan statistik

3. **doc/02_graph_construction.md**
   - Time-based graph grouping methodology
   - Haversine distance calculation
   - Tectonic penalty calculation
   - Prekursor window labeling (14 hari)
   - Multi-task label extraction (event, mag, azm, dist)
   - Handling missing data dengan zero-padding

4. **doc/03_physics_informed_loss.md**
   - Attenuation law: A = A₀ × e^(-α×d)
   - Multi-task loss function design
   - Focal loss untuk class imbalance
   - Angular loss untuk circular regression
   - Physics compliance metrics
   - Learnable attenuation coefficient

5. **doc/04_chronological_split.md**
   - Strict chronological split rationale
   - Train: ≤2023-12-31
   - Val: 2024-01-01 to 2025-03-31 (includes Kp=9.0 storm)
   - Test: ≥2026-01-01 (blind test)
   - Data leakage prevention
   - Class distribution analysis
   - Space weather analysis

6. **doc/05_model_architecture.md**
   - GAT encoder design (3 layers, 4 heads)
   - Multi-task output heads
   - Event detection head (binary classification)
   - Magnitude head (regression)
   - Azimuth head (circular regression with sin/cos)
   - Distance head (multi-output regression)
   - Global pooling strategy
   - Parameter count: ~550K
   - Training phases: Baseline → Physics Integration → Fine-tuning

### Scripts Implementasi

#### Data Processing Scripts
1. **scripts/01_data_cleaning.py**
   - Load dan clean lokasi_stasiun.csv
   - Load dan clean earthquake_catalog
   - Coordinate validation
   - Remove training artifacts
   - Generate cleaning report
   - Output: cleaned CSV files + log

2. **scripts/02_graph_construction.py**
   - Build adjacency matrix dengan Haversine distance
   - Calculate tectonic penalties
   - Create graph snapshots (one per day)
   - Extract multi-task labels
   - Handle missing data
   - Output: dataset_graphs.pt + metadata

3. **scripts/03_chronological_split.py**
   - Load all graph snapshots
   - Chronological split by date
   - Validate no temporal overlap
   - Analyze class distribution
   - Analyze magnitude distribution
   - Analyze space weather (val set)
   - Output: train/val/test graphs.pt + metadata

### Konfigurasi

#### config/model_config.yaml
- Model architecture parameters (GAT, DPINN)
- Loss function weights (λ₁-λ₅)
- Training hyperparameters
- Data augmentation settings
- Evaluation metrics
- Checkpoint strategy
- Logging configuration
- Hardware settings
- Cosmic Gating module (disabled by default)

### Project Files

#### Root Level
- **README.md**: Project overview, quick start, struktur
- **requirements.txt**: Python dependencies
- **CHANGELOG.md**: Version history dan changes
- **.gitignore**: Git ignore rules (data files, checkpoints, logs)

#### Data Directory
- **data/README.md**: Data structure, pipeline, quality checks

### Design Decisions Documented

#### 1. Multi-Station Simultaneous Processing
**Rationale**: Menghindari data leakage dengan memproses semua 24 stasiun dalam satu snapshot graf per hari, bukan per stasiun.

**Implementation**:
- Setiap graph snapshot = 1 hari × 24 stasiun
- Node features: spatial + seismic + status
- Edge features: distance + tectonic penalty

#### 2. Tectonic-Aware Adjacency
**Rationale**: Edge yang melintasi batas tektonik memiliki karakteristik propagasi yang berbeda.

**Implementation**:
- 3 klaster: Sunda (12), Wallacea (9), Sahul (3)
- Cross-tectonic penalty: 0.5
- Same-region penalty: 0.0

#### 3. Physics-Informed Loss
**Rationale**: Memastikan model mematuhi hukum fisika atenuasi seismik.

**Implementation**:
- Attenuation law: A = A₀ × e^(-α×d)
- Learnable α parameter (0.0001-0.001 km⁻¹)
- Physics loss: variance of A₀ estimates + MSE with expected amplitudes

#### 4. Strict Chronological Split
**Rationale**: Mencegah temporal data leakage dalam spatio-temporal prediction.

**Implementation**:
- Global time-based grouping
- No shuffling across time periods
- Train: past only (≤2023)
- Val: recent past (2024-2025)
- Test: future (≥2026)

#### 5. Multi-Task Learning
**Rationale**: Prediksi simultan multiple targets meningkatkan generalization.

**Implementation**:
- Event detection (binary)
- Magnitude prediction (regression)
- Azimuth prediction (circular regression)
- Distance prediction (multi-output regression)
- Weighted loss combination

### Technical Specifications

#### Dataset
- **Stations**: 24 seismik stations
- **Tectonic Regions**: Sunda, Wallacea, Sahul
- **Time Range**: 2018-2026+ (8+ years)
- **Prekursor Window**: 14 days
- **Significant Magnitude**: Mw ≥ 5.0

#### Graph Structure
- **Nodes**: 24 per snapshot
- **Edges**: 276 (fully connected, undirected)
- **Node Features**: 5 basic (lat, lon, elev, region, active)
- **Edge Features**: 3 (distance, penalty, weight)

#### Model
- **Architecture**: GAT + DPINN
- **GAT Layers**: 3
- **Attention Heads**: 4
- **Hidden Channels**: 128
- **Parameters**: ~550K
- **Output Tasks**: 4 (event, mag, azm, dist)

#### Training
- **Batch Size**: 32
- **Epochs**: 100
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Weights**: λ₁=1.0, λ₂=0.5, λ₃=0.3, λ₄=0.5, λ₅=0.2

### Status Saat Ini

#### ✅ Completed
- Project structure setup
- Complete documentation (6 documents)
- Data processing scripts (3 scripts)
- Model configuration file
- Project README and CHANGELOG
- .gitignore configuration

#### ⏳ Pending
- Raw data files placement
- Data cleaning execution
- Graph construction execution
- Chronological split execution
- Model implementation
- Training script
- Evaluation script
- Visualization tools
- Space weather data integration
- Cosmic Gating module

#### 🔍 Awaiting
- `data/raw/lokasi_stasiun.csv`
- `data/raw/earthquake_catalog_2018_2025_merged_robust.csv`

### Next Steps

1. **Immediate** (when data available):
   - Place raw data files in `data/raw/`
   - Run `scripts/01_data_cleaning.py`
   - Review cleaning logs
   - Run `scripts/02_graph_construction.py`
   - Review graph statistics
   - Run `scripts/03_chronological_split.py`
   - Verify split metadata

2. **Short-term**:
   - Implement model architecture
   - Create training script
   - Test with synthetic data
   - Validate physics loss

3. **Medium-term**:
   - Train on real data
   - Evaluate on validation set
   - Tune hyperparameters
   - Add visualization tools

4. **Long-term**:
   - Blind test evaluation
   - Space weather integration
   - Cosmic Gating module
   - Production deployment

### Notes

- Semua dokumentasi dalam Bahasa Indonesia sesuai permintaan
- Strict adherence to chronological split untuk menghindari data leakage
- Physics-informed approach untuk meningkatkan interpretability
- Multi-task learning untuk robust prediction
- Modular design untuk easy extension

### References Added

1. Kipf & Welling (2017) - Graph Convolutional Networks
2. Veličković et al. (2018) - Graph Attention Networks
3. Raissi et al. (2019) - Physics-Informed Neural Networks
4. Wu et al. (2020) - Spatio-Temporal Graph Neural Networks
5. Battaglia et al. (2018) - Relational Inductive Biases

---

**Date**: 2026-05-02  
**Phase**: Initialization Complete  
**Next Milestone**: Data Processing Pipeline Execution
