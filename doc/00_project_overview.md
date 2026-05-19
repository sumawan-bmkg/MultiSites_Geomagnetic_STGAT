# ANTIGRAVITY Project Overview

**Tanggal Dibuat**: 2 Mei 2026  
**Status**: Initialization Phase

## Tujuan Proyek

Proyek ANTIGRAVITY bertujuan untuk mengembangkan sistem prediksi gempa bumi menggunakan kombinasi:
1. **Spatio-Temporal Graph Neural Network (GNN)** - untuk menangkap pola spasial antar stasiun
2. **Dynamic Physics-Informed Neural Network (DPINN)** - untuk mengintegrasikan hukum fisika atenuasi seismik

## Dataset

### Sumber Data
- **lokasi_stasiun.csv**: Koordinat dan metadata 24 stasiun seismik di Indonesia
- **earthquake_catalog_2018_2025_merged_robust.csv**: Katalog gempa bumi 2018-2025

### Stasiun Seismik (24 Stasiun)

#### Klaster 0 - Sunda (12 stasiun)
- SBG, MLB, SCN, KPY, LWA, LPS, SRG, SKB, CLP, YOG, TRT, GSI

#### Klaster 1 - Wallacea (9 stasiun)
- TND, GTO, LWK, PLU, TNT, TRD, LUT, ALR, ROT

#### Klaster 2 - Sahul (3 stasiun)
- SMI, AMB, JYP, SRO

## Arsitektur Model

### Graph Neural Network (GNN)
- **Node Features**: Fitur spatio-temporal dari setiap stasiun
- **Edge Features**: Jarak Haversine + penalti tektonik
- **Aggregation**: Graph Attention Network (GAT)

### Physics-Informed Neural Network (DPINN)
- **Physics Loss**: Atenuasi amplitudo seismik
  ```
  A = A₀ × e^(-α × d)
  ```
  dimana:
  - A = amplitudo pada jarak d
  - A₀ = amplitudo awal
  - α = koefisien atenuasi
  - d = jarak dari episenter (km)

## Multi-Task Learning

Model memprediksi 4 target secara simultan:

1. **label_event** (Binary): 1 jika dalam jendela prekursor, 0 jika background noise
2. **label_mag** (Regression): Magnitudo gempa (Mw)
3. **label_azm** (Regression): Azimuth dari pusat graf ke episenter (derajat)
4. **label_dist** (Regression): Jarak dari stasiun ke episenter (km)

## Pembagian Data (Strict Chronological Split)

### Training Set
- **Periode**: Awal observasi - 31 Desember 2023
- **Tujuan**: Pembelajaran pola prekursor gempa

### Validation Set
- **Periode**: 1 Januari 2024 - 31 Maret 2025
- **Tujuan**: Hyperparameter tuning dan evaluasi Cosmic Gating
- **Catatan**: Termasuk anomali badai matahari Kp=9.0 (Mei 2024)

### Test Set (Blind Test)
- **Periode**: 1 Januari 2026 - seterusnya
- **Tujuan**: Evaluasi final pada data yang benar-benar unseen

## Pencegahan Data Leakage

### Prinsip Utama
1. **Global Time-Based Grouping**: Setiap snapshot graf mewakili SATU HARI untuk SEMUA 24 stasiun
2. **No Future Information**: Data validation/test tidak boleh mempengaruhi training
3. **Strict Chronological Order**: Tidak ada shuffling antar periode waktu

### Handling Missing Data
- **Zero-padding**: Untuk stasiun yang tidak aktif pada hari tertentu
- **Masking tensor**: Memberitahu GAT bahwa node tersebut tidak aktif

## Inovasi Utama

1. **Multi-Station Simultaneous Processing**: Tidak memproses stasiun secara independen
2. **Tectonic-Aware Adjacency**: Penalti untuk edge yang melintasi batas tektonik
3. **Physics-Guided Loss**: Mengintegrasikan hukum fisika dalam training
4. **Cosmic Gating Module**: Menangani pengaruh space weather (Kp, Dst)

## Metrik Evaluasi

### Classification Metrics (Event Detection)
- Precision, Recall, F1-Score
- ROC-AUC

### Regression Metrics
- MAE (Mean Absolute Error) untuk magnitude
- RMSE (Root Mean Square Error) untuk distance
- Angular Error untuk azimuth

### Physics Compliance
- Physics Loss: Seberapa baik model mematuhi hukum atenuasi

## Timeline Proyek

1. **Phase 1**: Data Cleaning & Validation ✓ (Dokumen ini)
2. **Phase 2**: Graph Construction
3. **Phase 3**: Model Development
4. **Phase 4**: Training & Validation
5. **Phase 5**: Blind Test & Deployment

## Referensi

1. Kipf & Welling (2017) - Graph Convolutional Networks
2. Veličković et al. (2018) - Graph Attention Networks
3. Raissi et al. (2019) - Physics-Informed Neural Networks
4. Wu et al. (2020) - Spatio-Temporal Graph Neural Networks

## Catatan Penting

⚠️ **CRITICAL**: Folder "intial" dengan file data belum ditemukan di workspace. Pastikan file berikut tersedia:
- `data/raw/lokasi_stasiun.csv`
- `data/raw/earthquake_catalog_2018_2025_merged_robust.csv`

Setelah file tersedia, jalankan script preprocessing secara berurutan.
