# ANTIGRAVITY Project

## Spatio-Temporal Graph Neural Network (GNN) & Dynamic Physics-Informed Neural Network (DPINN)

Proyek ini bertujuan untuk melakukan prediksi gempa bumi menggunakan kombinasi Graph Neural Network (GNN) dan Physics-Informed Neural Network (DPINN) dengan data dari 24 stasiun seismik di Indonesia.

## Struktur Proyek

```
ANTIGRAVITY/
├── data/               # Dataset dan file data
│   ├── raw/           # Data mentah (lokasi_stasiun.csv, earthquake_catalog)
│   ├── processed/     # Data yang sudah diproses
│   ├── train/         # Data training (hingga 31 Des 2023)
│   ├── val/           # Data validation (1 Jan 2024 - 31 Mar 2025)
│   └── test/          # Data test (mulai 1 Jan 2026)
├── models/            # Model arsitektur dan checkpoint
├── scripts/           # Script pemrosesan dan training
├── config/            # File konfigurasi
├── plots/             # Visualisasi dan grafik
├── references/        # Paper dan referensi penelitian
├── doc/               # Dokumentasi lengkap
└── notebooks/         # Jupyter notebooks untuk eksplorasi

```

## Fitur Utama

1. **Multi-Station Graph Processing**: Memproses 24 stasiun secara simultan dalam satu snapshot graf
2. **Physics-Informed Loss**: Menggunakan atenuasi amplitudo seismik A = A₀e^(-αd)
3. **Tectonic Clustering**: Klasterisasi stasiun berdasarkan region tektonik (Sunda, Wallacea, Sahul)
4. **Strict Chronological Split**: Pembagian data yang ketat berdasarkan waktu untuk menghindari data leakage
5. **Multi-Task Learning**: Prediksi event, magnitude, azimuth, dan distance secara simultan

## Instalasi

```bash
pip install -r requirements.txt
```

## Quick Start

1. Letakkan data mentah di folder `data/raw/`:
   - `lokasi_stasiun.csv`
   - `earthquake_catalog_2018_2025_merged_robust.csv`

2. Jalankan preprocessing:
```bash
python scripts/01_data_cleaning.py
python scripts/02_graph_construction.py
python scripts/03_chronological_split.py
```

3. Training model:
```bash
python scripts/04_train_model.py
```

## Dokumentasi

Lihat folder `doc/` untuk dokumentasi lengkap setiap tahap:
- [00_project_overview.md](doc/00_project_overview.md)
- [01_data_cleaning.md](doc/01_data_cleaning.md)
- [02_graph_construction.md](doc/02_graph_construction.md)
- [03_physics_informed_loss.md](doc/03_physics_informed_loss.md)
- [04_chronological_split.md](doc/04_chronological_split.md)

## Lisensi

MIT License

## Kontak

ANTIGRAVITY Research Team
