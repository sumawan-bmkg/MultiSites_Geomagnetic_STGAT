# Multi-Site STGAT Resilient V2
### Spatio-Temporal Graph Attention Network with Geodesic Prior & Unit Circle Cosine Similarity Loss

<img width="1024" height="440" alt="graphical_abstract" src="https://github.com/user-attachments/assets/ae7135b8-3362-425d-933c-075607b7b607" />


Proyek ini merupakan repositori resmi untuk arsitektur deep learning **Multi-Site STGAT Resilient V2**, sebuah sistem deteksi anomali Pc3/Pc5 geomagnetic terdistribusi berbasis **Spatio-Temporal Graph Attention Network** yang dirancang untuk melakukan prakiraan parameter gempa bumi secara terdistribusi di 24 stasiun magnetometer seismik BMKG Indonesia.

---

## 🚀 Fitur Unggulan SOTA (V2 Resilient Edition)

1. **Geodesic Prior Matrix Injection (Strategi C):**
   Mengintegrasikan peta koordinat GPS stasiun bumi melalui matriks prior jarak spasial geodesik $24 \times 24$ ke dalam Graph Attention Network (GAT) spasial. Ini membiaskan atensi dinamis secara fisik berdasarkan hukum propagasi fisis bumi untuk akurasi pelokalan azimuth stasiun yang sangat tinggi.
2. **Unit Vector Cosine Similarity Loss (Strategi A):**
   Memproyeksikan target derajat azimuth ($0^\circ - 360^\circ$) ke dalam sistem koordinat Kartesian Unit Circle 2D. Dengan membandingkan vektor prediksi dan target via Cosine Loss, model berhasil menembus diskontinuitas batas lingkaran derajat tradisional sehingga mencapai konvergensi MAE Azimuth optimal.
3. **Hard Physics Gate (Kp-Index Suppression):**
   Filter deterministik hibrida AI-Fisika pada loop inferensi yang memanfaatkan fitur indeks matahari (Kp dan Dst) dari `cosmic_features` untuk memblokir anomali semu (False Positives) selama periode badai geomagnetik matahari ekstrem (Solar Cycle 25).
4. **Phase-Preserving Temporal Max Pooling:**
   Menggunakan Max Pooling pada ekstraksi fitur temporal guna mempertahankan fase puncak kedatangan gelombang mikro Pc3/Pc5 yang krusial bagi triangulasi spasial, menggantikan model rata-rata (Average Pooling) konvensional.


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

Geomagnetic Research Team
