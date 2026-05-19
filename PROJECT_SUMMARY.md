# ANTIGRAVITY Project - Summary

**Date**: 2 Mei 2026  
**Status**: ✅ Initialization Complete - Awaiting Data

---

## 🎯 Tujuan Proyek

Mengembangkan sistem prediksi gempa bumi menggunakan **Spatio-Temporal Graph Neural Network (GNN)** dan **Dynamic Physics-Informed Neural Network (DPINN)** dengan data dari 24 stasiun seismik di Indonesia.

## 📊 Dataset

### Stasiun Seismik (24 Stasiun)
- **Klaster Sunda** (12): SBG, MLB, SCN, KPY, LWA, LPS, SRG, SKB, CLP, YOG, TRT, GSI
- **Klaster Wallacea** (9): TND, GTO, LWK, PLU, TNT, TRD, LUT, ALR, ROT
- **Klaster Sahul** (3): SMI, AMB, JYP, SRO

### Data Requirements
📁 **Diperlukan di `data/raw/`**:
1. `lokasi_stasiun.csv` - Koordinat 24 stasiun
2. `earthquake_catalog_2018_2025_merged_robust.csv` - Katalog gempa 2018-2025

## 🏗️ Struktur Proyek

```
ANTIGRAVITY/
├── 📄 README.md                    # Project overview
├── 📄 CHANGELOG.md                 # Version history
├── 📄 PROJECT_SUMMARY.md           # This file
├── 📄 requirements.txt             # Dependencies
├── 📄 .gitignore                   # Git ignore rules
│
├── 📁 data/                        # Data storage
│   ├── raw/                        # ⚠️ Place raw data here
│   ├── processed/                  # Cleaned data
│   ├── train/                      # Training set (≤2023)
│   ├── val/                        # Validation set (2024-2025)
│   └── test/                       # Test set (≥2026)
│
├── 📁 doc/                         # Documentation
│   ├── 00_project_overview.md     # ✅ Project goals & design
│   ├── 01_data_cleaning.md        # ✅ Data cleaning procedures
│   ├── 02_graph_construction.md   # ✅ Graph building methodology
│   ├── 03_physics_informed_loss.md # ✅ Physics loss design
│   ├── 04_chronological_split.md  # ✅ Data split strategy
│   ├── 05_model_architecture.md   # ✅ Model design
│   └── CHANGELOG_PROJECT.md       # ✅ Development log
│
├── 📁 scripts/                     # Processing scripts
│   ├── 01_data_cleaning.py        # ✅ Clean raw data
│   ├── 02_graph_construction.py   # ✅ Build graph snapshots
│   └── 03_chronological_split.py  # ✅ Split dataset
│
├── 📁 config/                      # Configuration
│   └── model_config.yaml          # ✅ Model & training config
│
├── 📁 models/                      # Model storage
├── 📁 plots/                       # Visualizations
├── 📁 references/                  # Research papers
└── 📁 notebooks/                   # Jupyter notebooks
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
# Place raw data files in data/raw/
# - lokasi_stasiun.csv
# - earthquake_catalog_2018_2025_merged_robust.csv
```

### 3. Run Data Pipeline
```bash
# Step 1: Clean data
python scripts/01_data_cleaning.py

# Step 2: Build graphs
python scripts/02_graph_construction.py

# Step 3: Split dataset
python scripts/03_chronological_split.py
```

### 4. Review Outputs
```bash
# Check logs
cat data/processed/cleaning_log.txt
cat data/processed/graph_construction_log.txt
cat data/processed/chronological_split_log.txt

# Check metadata
cat data/processed/dataset_metadata.json
cat data/processed/split_metadata.json
```

## 🔬 Metodologi

### 1. Multi-Station Graph Processing
- **Konsep**: Setiap snapshot graf = 1 hari × 24 stasiun
- **Tujuan**: Menghindari spatio-temporal data leakage
- **Output**: One graph per day dengan 24 nodes

### 2. Tectonic-Aware Adjacency
- **Edge Penalty**: 0.5 untuk cross-tectonic, 0.0 untuk same-region
- **Distance**: Haversine formula untuk jarak geografis
- **Connectivity**: Fully connected (276 edges per graph)

### 3. Physics-Informed Loss
- **Formula**: A = A₀ × e^(-α×d)
- **Parameter**: α (learnable attenuation coefficient)
- **Range**: 0.0001 - 0.001 km⁻¹

### 4. Strict Chronological Split
- **Train**: Hingga 31 Des 2023 (~2,190 hari)
- **Val**: 1 Jan 2024 - 31 Mar 2025 (~455 hari)
- **Test**: Mulai 1 Jan 2026 (~120+ hari)
- **Rationale**: Mencegah temporal data leakage

### 5. Multi-Task Learning
1. **Event Detection**: Binary (prekursor vs background)
2. **Magnitude**: Regression (Mw)
3. **Azimuth**: Circular regression (0-360°)
4. **Distance**: Multi-output regression (24 distances)

## 🧠 Model Architecture

### Graph Attention Network (GAT)
- **Layers**: 3
- **Hidden Channels**: 128
- **Attention Heads**: 4
- **Dropout**: 0.3
- **Parameters**: ~500K

### Output Heads
- **Event Head**: Binary classification
- **Magnitude Head**: Regression
- **Azimuth Head**: Circular regression (sin/cos)
- **Distance Head**: Multi-output regression (24 outputs)

### Total Parameters
~550K parameters

## 📈 Expected Performance

### Dataset Statistics (Estimated)
- **Total Graphs**: ~2,800 (8 years × 365 days)
- **Event Ratio**: 5-15%
- **Imbalance**: 1:6 to 1:20
- **Magnitude Range**: 5.0 - 8.0+

### Evaluation Metrics
- **Classification**: Precision, Recall, F1, ROC-AUC
- **Regression**: MAE, RMSE, R²
- **Circular**: Angular Error
- **Physics**: Compliance Error, Learned α

## ✅ Completed Tasks

### Documentation (6 files)
- [x] Project overview
- [x] Data cleaning procedures
- [x] Graph construction methodology
- [x] Physics-informed loss design
- [x] Chronological split strategy
- [x] Model architecture design

### Scripts (3 files)
- [x] Data cleaning script
- [x] Graph construction script
- [x] Chronological split script

### Configuration
- [x] Model config YAML
- [x] Requirements.txt
- [x] .gitignore

### Project Files
- [x] README.md
- [x] CHANGELOG.md
- [x] PROJECT_SUMMARY.md
- [x] data/README.md

## ⏳ Pending Tasks

### Immediate (Awaiting Data)
- [ ] Place raw data files
- [ ] Execute data cleaning
- [ ] Execute graph construction
- [ ] Execute chronological split

### Short-term
- [ ] Implement model architecture
- [ ] Create training script
- [ ] Create evaluation script
- [ ] Test with synthetic data

### Medium-term
- [ ] Train on real data
- [ ] Hyperparameter tuning
- [ ] Visualization tools
- [ ] Performance analysis

### Long-term
- [ ] Blind test evaluation
- [ ] Space weather integration
- [ ] Cosmic Gating module
- [ ] Production deployment

## 🔍 Key Features

### 1. No Data Leakage
✅ Strict chronological split  
✅ Global time-based grouping  
✅ No future information in training

### 2. Physics-Guided
✅ Attenuation law integration  
✅ Learnable physics parameters  
✅ Physics compliance metrics

### 3. Multi-Task Learning
✅ Simultaneous predictions  
✅ Shared representations  
✅ Improved generalization

### 4. Tectonic-Aware
✅ Regional clustering  
✅ Cross-boundary penalties  
✅ Geologically informed

### 5. Scalable & Modular
✅ Easy to extend  
✅ Well-documented  
✅ Reproducible

## 📚 Documentation

### Main Documents
1. **doc/00_project_overview.md** - Start here for project understanding
2. **doc/01_data_cleaning.md** - Data preparation details
3. **doc/02_graph_construction.md** - Graph building methodology
4. **doc/03_physics_informed_loss.md** - Physics integration
5. **doc/04_chronological_split.md** - Data split strategy
6. **doc/05_model_architecture.md** - Model design details

### Additional Resources
- **README.md** - Quick start guide
- **CHANGELOG.md** - Version history
- **data/README.md** - Data directory guide
- **doc/CHANGELOG_PROJECT.md** - Development log

## 🛠️ Dependencies

### Core
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- PyTorch Geometric ≥ 2.3.0

### Data Processing
- pandas ≥ 2.0.0
- numpy ≥ 1.24.0
- scipy ≥ 1.10.0
- h5py ≥ 3.8.0

### Visualization
- matplotlib ≥ 3.7.0
- seaborn ≥ 0.12.0
- plotly ≥ 5.14.0

### Utilities
- scikit-learn ≥ 1.3.0
- tqdm ≥ 4.65.0
- pyyaml ≥ 6.0

## 🎓 References

1. Kipf & Welling (2017) - Graph Convolutional Networks
2. Veličković et al. (2018) - Graph Attention Networks
3. Raissi et al. (2019) - Physics-Informed Neural Networks
4. Wu et al. (2020) - Spatio-Temporal Graph Neural Networks

## ⚠️ Important Notes

### Data Files
- Raw data files are **NOT** tracked in git (too large)
- Must be placed manually in `data/raw/` directory
- Processed files can be regenerated from raw data

### Chronological Split
- **CRITICAL**: Never shuffle data across time periods
- Train/val/test split is **strictly chronological**
- This prevents temporal data leakage

### Physics Loss
- Attenuation coefficient α should converge to 0.0001-0.001 km⁻¹
- If α is outside this range, check physics loss implementation
- Physics loss weight (λ₅) should be tuned carefully

### Space Weather
- Validation set includes May 2024 Kp=9.0 storm
- Use this to evaluate Cosmic Gating module
- Space weather data integration is optional but recommended

## 📞 Next Actions

### For Users
1. ✅ Review documentation in `doc/` folder
2. ⏳ Place raw data files in `data/raw/`
3. ⏳ Run data processing pipeline
4. ⏳ Review logs and metadata
5. ⏳ Proceed to model training

### For Developers
1. ✅ Project structure complete
2. ✅ Documentation complete
3. ✅ Scripts ready
4. ⏳ Implement model architecture
5. ⏳ Create training pipeline

---

**Project Status**: 🟢 Ready for Data Processing  
**Last Updated**: 2026-05-02  
**Version**: 0.1.0

**Contact**: ANTIGRAVITY Research Team
