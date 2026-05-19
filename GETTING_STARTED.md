# Getting Started with ANTIGRAVITY

Panduan lengkap untuk memulai proyek ANTIGRAVITY dari awal.

---

## 📋 Prerequisites

### System Requirements
- **OS**: Windows, Linux, atau macOS
- **Python**: 3.8 atau lebih tinggi
- **RAM**: Minimum 8GB (16GB recommended)
- **GPU**: Optional (CUDA-compatible untuk training lebih cepat)
- **Storage**: Minimum 10GB free space

### Required Knowledge
- Python programming
- Basic understanding of:
  - Graph Neural Networks
  - PyTorch
  - Seismology (helpful but not required)

---

## 🚀 Step-by-Step Setup

### Step 1: Clone or Download Project

Jika menggunakan git:
```bash
git clone <repository-url>
cd ANTIGRAVITY
```

Atau download dan extract ZIP file.

### Step 2: Create Virtual Environment

**Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: PyTorch Geometric installation might require additional steps. See [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

Expected output:
```
PyTorch: 2.0.0+cu118 (or similar)
PyG: 2.3.0 (or similar)
```

---

## 📁 Prepare Data

### Step 5: Place Raw Data Files

Letakkan file data mentah di folder `data/raw/`:

1. **lokasi_stasiun.csv**
   - Format: CSV dengan kolom `Kode Stasiun`, `Latitude`, `Longitude`, `Elevation`
   - Expected: 24 baris (24 stasiun)

2. **earthquake_catalog_2018_2025_merged_robust.csv**
   - Format: CSV dengan kolom `time`, `latitude`, `longitude`, `depth`, `mag`
   - Expected: 10,000-50,000 baris (tergantung threshold)

**Struktur yang diharapkan**:
```
data/
└── raw/
    ├── lokasi_stasiun.csv
    └── earthquake_catalog_2018_2025_merged_robust.csv
```

### Step 6: Verify Data Files

```bash
# Windows
dir data\raw

# Linux/macOS
ls -lh data/raw/
```

Pastikan kedua file ada dan ukurannya reasonable (bukan 0 bytes).

---

## 🔧 Data Processing Pipeline

### Step 7: Data Cleaning

```bash
python scripts/01_data_cleaning.py
```

**Expected output**:
- `data/processed/lokasi_stasiun_clean.csv`
- `data/processed/earthquake_catalog_clean.csv`
- `data/processed/earthquake_catalog_significant.csv`
- `data/processed/cleaning_log.txt`

**Verification**:
```bash
# Check log
cat data/processed/cleaning_log.txt

# Or on Windows
type data\processed\cleaning_log.txt
```

Look for:
- ✓ Station count: 24
- ✓ No errors in coordinate validation
- ✓ Earthquake catalog cleaned successfully

### Step 8: Graph Construction

```bash
python scripts/02_graph_construction.py
```

**Expected output**:
- `data/processed/dataset_graphs.pt`
- `data/processed/dataset_metadata.json`
- `data/processed/graph_construction_log.txt`

**Verification**:
```bash
# Check metadata
cat data/processed/dataset_metadata.json

# Or on Windows
type data\processed\dataset_metadata.json
```

Look for:
- `num_graphs`: ~2,800 (8 years × 365 days)
- `num_stations`: 24
- `event_count`: Should be > 0

### Step 9: Chronological Split

```bash
python scripts/03_chronological_split.py
```

**Expected output**:
- `data/train/graphs.pt`
- `data/val/graphs.pt`
- `data/test/graphs.pt`
- `data/processed/split_metadata.json`
- `data/processed/chronological_split_log.txt`

**Verification**:
```bash
# Check split metadata
cat data/processed/split_metadata.json

# Or on Windows
type data\processed\split_metadata.json
```

Look for:
- Train count: ~2,190 graphs
- Val count: ~455 graphs
- Test count: ~120+ graphs
- No temporal overlap

---

## 📊 Explore Data (Optional)

### Step 10: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `notebooks/` directory and create a new notebook to explore:

```python
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load station data
stations = pd.read_csv('data/processed/lokasi_stasiun_clean.csv')
print(stations.head())

# Load graphs
graphs = torch.load('data/train/graphs.pt')
print(f"Training graphs: {len(graphs)}")
print(f"First graph: {graphs[0]}")

# Plot station distribution
plt.figure(figsize=(10, 6))
plt.scatter(stations['Longitude'], stations['Latitude'], 
           c=stations['region_idx'], cmap='viridis', s=100)
plt.colorbar(label='Region Index')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('24 Seismic Stations')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🧠 Model Development (Coming Soon)

### Step 11: Implement Model Architecture

File: `models/architecture.py` (to be created)

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class ANTIGRAVITYModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Implementation based on doc/05_model_architecture.md
        pass
    
    def forward(self, data):
        # Forward pass
        pass
```

### Step 12: Create Training Script

File: `scripts/04_train_model.py` (to be created)

```python
import torch
from torch_geometric.loader import DataLoader
from models.architecture import ANTIGRAVITYModel
from models.loss import MultiTaskLoss

# Load config
# Load data
# Initialize model
# Training loop
```

### Step 13: Train Model

```bash
python scripts/04_train_model.py --config config/model_config.yaml
```

---

## 📈 Evaluation (Coming Soon)

### Step 14: Evaluate on Test Set

```bash
python scripts/05_evaluate_model.py --checkpoint models/checkpoints/best_model.pt
```

### Step 15: Generate Visualizations

```bash
python scripts/06_visualize_results.py
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "File not found" error
**Problem**: Raw data files tidak ada di `data/raw/`

**Solution**:
```bash
# Verify files exist
ls data/raw/

# Check file names (case-sensitive!)
# Should be exactly:
# - lokasi_stasiun.csv
# - earthquake_catalog_2018_2025_merged_robust.csv
```

#### 2. "Module not found" error
**Problem**: Dependencies tidak terinstall

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific package
pip install torch-geometric
```

#### 3. PyTorch Geometric installation fails
**Problem**: PyG memerlukan instalasi khusus

**Solution**:
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Install PyG for your PyTorch version
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# Example for PyTorch 2.0.0 + CUDA 11.8:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

#### 4. "Expected 24 stations, got X"
**Problem**: Station data tidak lengkap atau ada duplikat

**Solution**:
```bash
# Check cleaning log
cat data/processed/cleaning_log.txt

# Review which stations were removed and why
# Fix raw data file if needed
```

#### 5. Out of memory error
**Problem**: GPU/RAM tidak cukup

**Solution**:
```yaml
# Edit config/model_config.yaml
training:
  batch_size: 16  # Reduce from 32
  
hardware:
  device: "cpu"  # Use CPU instead of GPU
```

#### 6. "No events in validation set"
**Problem**: Earthquake catalog tidak mencakup periode 2024-2025

**Solution**:
```bash
# Check earthquake date range
python -c "import pandas as pd; df = pd.read_csv('data/processed/earthquake_catalog_clean.csv'); print(df['time'].min(), df['time'].max())"

# Verify significant events
python -c "import pandas as pd; df = pd.read_csv('data/processed/earthquake_catalog_significant.csv'); print(len(df))"
```

---

## 📚 Next Steps

After completing setup:

1. **Read Documentation**:
   - `doc/00_project_overview.md` - Understand project goals
   - `doc/01_data_cleaning.md` - Data cleaning details
   - `doc/02_graph_construction.md` - Graph methodology
   - `doc/03_physics_informed_loss.md` - Physics integration
   - `doc/04_chronological_split.md` - Data split strategy
   - `doc/05_model_architecture.md` - Model design

2. **Explore Data**:
   - Use Jupyter notebooks in `notebooks/`
   - Visualize station distribution
   - Analyze earthquake patterns
   - Examine graph structure

3. **Implement Model**:
   - Follow `doc/05_model_architecture.md`
   - Implement GAT encoder
   - Implement multi-task heads
   - Implement physics loss

4. **Train & Evaluate**:
   - Train on training set
   - Validate on validation set
   - Test on blind test set
   - Analyze results

---

## 💡 Tips

### Performance Optimization
- Use GPU if available (10-50x faster)
- Adjust batch size based on available memory
- Use mixed precision training (FP16)
- Profile code to find bottlenecks

### Best Practices
- Always check logs after each step
- Verify data quality before training
- Save checkpoints regularly
- Document experiments
- Version control your code

### Learning Resources
- PyTorch Geometric tutorials: https://pytorch-geometric.readthedocs.io/
- Graph Neural Networks course: https://web.stanford.edu/class/cs224w/
- Physics-Informed Neural Networks: https://maziarraissi.github.io/PINNs/

---

## 🆘 Getting Help

### Documentation
- Check `doc/` folder for detailed documentation
- Read `README.md` files in each directory
- Review `CHANGELOG.md` for recent changes

### Logs
- Always check log files in `data/processed/`
- Look for error messages and warnings
- Verify statistics match expectations

### Community
- Open an issue on GitHub (if applicable)
- Check existing issues for similar problems
- Provide logs and error messages when asking for help

---

## ✅ Checklist

Before proceeding to model training:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Raw data files in `data/raw/`
- [ ] Data cleaning completed successfully
- [ ] Graph construction completed successfully
- [ ] Chronological split completed successfully
- [ ] All logs reviewed and no errors
- [ ] Metadata files verified
- [ ] Data exploration completed (optional)

---

**Ready to start?** Begin with Step 1 above! 🚀

**Questions?** Check documentation in `doc/` folder or review logs in `data/processed/`.

**Good luck with your earthquake prediction research!** 🌍
