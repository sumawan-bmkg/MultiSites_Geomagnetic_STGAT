# Tahap 1: Data Cleaning & Validation

**Tanggal**: 2 Mei 2026  
**Status**: Script Ready - Menunggu Data

## Tujuan

Membersihkan dan memvalidasi data mentah dari:
1. `lokasi_stasiun.csv` - Koordinat 24 stasiun seismik
2. `earthquake_catalog_2018_2025_merged_robust.csv` - Katalog gempa

## Langkah-Langkah

### 1.1 Validasi lokasi_stasiun.csv

#### Input Expected
```csv
Kode Stasiun,Latitude,Longitude,Elevation,Region
SBG,-6.8995,107.6076,700,Sunda
MLB,-7.9778,112.6347,450,Sunda
...
```

#### Proses Cleaning
1. **Load Data**: Baca CSV dengan pandas
2. **Check Missing Values**: 
   - Hapus baris dengan NaN pada kolom: `Kode Stasiun`, `Latitude`, `Longitude`
   - Elevation boleh NaN (akan diisi dengan 0)
3. **Validate Count**: Pastikan tepat 24 stasiun valid
4. **Coordinate Validation**:
   - Latitude: -11° hingga 6° (Indonesia)
   - Longitude: 95° hingga 141° (Indonesia)
5. **Duplicate Check**: Pastikan tidak ada Kode Stasiun duplikat

#### Output
- `data/processed/lokasi_stasiun_clean.csv`
- Log: Jumlah baris dihapus, alasan penghapusan

### 1.2 Validasi earthquake_catalog_2018_2025_merged_robust.csv

#### Input Expected
```csv
time,latitude,longitude,depth,mag,magType,place,...
2018-01-01T00:15:23.000Z,-7.5,110.2,10.0,5.2,Mw,Java,...
```

#### Proses Cleaning
1. **Load Data**: Baca CSV dengan pandas
2. **DateTime Conversion**:
   ```python
   df['time'] = pd.to_datetime(df['time'], utc=True)
   ```
3. **Remove Anomalies**:
   - Hapus baris dengan kolom `station` berisi string angka (e.g., '20240125')
   - Ini adalah bug penamaan dari training historis
4. **Validate Required Columns**:
   - `time`: datetime valid
   - `latitude`, `longitude`: koordinat valid
   - `depth`: > 0 km
   - `mag`: magnitude valid (biasanya 0-10)
5. **Filter by Magnitude**:
   - Untuk prekursor analysis, fokus pada Mw ≥ 5.0
   - Simpan juga event kecil untuk background noise
6. **Coordinate Validation**:
   - Pastikan gempa dalam region Indonesia atau sekitarnya
   - Latitude: -15° hingga 10°
   - Longitude: 90° hingga 145°

#### Output
- `data/processed/earthquake_catalog_clean.csv`
- `data/processed/earthquake_catalog_significant.csv` (Mw ≥ 5.0)
- Log: Statistik cleaning

### 1.3 Metadata Training Historis

#### Identifikasi Bug
Beberapa baris dalam katalog gempa mungkin memiliki kolom `station` yang berisi:
- String angka: '20240125', '20231215', dll.
- Ini adalah timestamp yang salah masuk ke kolom station

#### Cleaning Strategy
```python
# Hapus baris dengan station berisi hanya angka
df = df[~df['station'].str.match(r'^\d+$', na=False)]
```

## Validasi Output

### Checklist
- [ ] lokasi_stasiun_clean.csv memiliki tepat 24 baris
- [ ] Tidak ada NaN pada koordinat stasiun
- [ ] earthquake_catalog_clean.csv tidak memiliki anomali station
- [ ] Semua timestamp dalam format datetime standar
- [ ] Koordinat semua event dalam range Indonesia

### Statistik yang Diharapkan

#### Stasiun
- Total stasiun valid: 24
- Klaster Sunda: 12 stasiun
- Klaster Wallacea: 9 stasiun
- Klaster Sahul: 3 stasiun

#### Gempa
- Total events (2018-2025): ~10,000-50,000 (tergantung threshold)
- Significant events (Mw ≥ 5.0): ~500-2,000
- Range magnitude: 0.0 - 8.0+
- Range depth: 0 - 700 km

## Script Implementasi

Script: `scripts/01_data_cleaning.py`

### Fungsi Utama
1. `clean_station_data()`: Membersihkan lokasi_stasiun.csv
2. `clean_earthquake_catalog()`: Membersihkan earthquake_catalog
3. `validate_coordinates()`: Validasi koordinat geografis
4. `remove_training_artifacts()`: Hapus bug metadata training
5. `generate_cleaning_report()`: Buat laporan cleaning

### Logging
Semua operasi cleaning dicatat dalam:
- `data/processed/cleaning_log.txt`
- Format: timestamp, operasi, jumlah baris affected, alasan

## Error Handling

### Potential Issues
1. **File Not Found**: Jika file tidak ada di `data/raw/`
   - Action: Print error message, exit gracefully
2. **Encoding Issues**: File CSV dengan encoding non-UTF8
   - Action: Try multiple encodings (utf-8, latin-1, cp1252)
3. **Missing Columns**: Kolom expected tidak ada
   - Action: Print missing columns, exit dengan error message
4. **All Data Invalid**: Semua baris gagal validasi
   - Action: Print warning, review validation criteria

## Next Steps

Setelah data cleaning selesai:
1. Review `cleaning_log.txt` untuk memastikan tidak ada data penting yang hilang
2. Visualisasi distribusi stasiun dan gempa (optional)
3. Lanjut ke **Tahap 2: Graph Construction**

## Changelog

- **2026-05-02**: Dokumentasi initial dibuat
- **Pending**: Eksekusi cleaning setelah data tersedia
