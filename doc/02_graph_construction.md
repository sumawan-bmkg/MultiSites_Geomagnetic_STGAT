# Tahap 2: Graph Construction - Time-Based Multi-Station Snapshots

**Tanggal**: 2 Mei 2026  
**Status**: Design Phase

## Tujuan

Membuat **Snapshot Graf** untuk setiap hari observasi, dimana setiap snapshot berisi:
- 24 nodes (satu untuk setiap stasiun)
- Edge connections berdasarkan jarak geografis dan penalti tektonik
- Node features dari data seismik/prekursor
- Labels dari katalog gempa (jika ada event dalam jendela prekursor)

## Konsep Utama

### Time-Based Grouping (KRITIS untuk Menghindari Data Leakage)

**SALAH** ❌:
```
Stasiun A: [day1, day2, day3, ...] → shuffle → train/val/test
Stasiun B: [day1, day2, day3, ...] → shuffle → train/val/test
```
Ini menyebabkan data leakage karena hari yang sama bisa masuk ke train dan test.

**BENAR** ✅:
```
Day 1: Graph(24 stasiun) → train
Day 2: Graph(24 stasiun) → train
...
Day N: Graph(24 stasiun) → val
...
Day M: Graph(24 stasiun) → test
```

## Struktur Graph Snapshot

### Node Representation
Setiap node mewakili satu stasiun pada hari tertentu:

```python
node_features = {
    'station_id': int,           # 0-23
    'latitude': float,
    'longitude': float,
    'elevation': float,
    'region_idx': int,           # 0=Sunda, 1=Wallacea, 2=Sahul
    'is_active': bool,           # True jika ada data, False jika missing
    
    # Fitur seismik (jika tersedia)
    'amplitude': float,          # Zero-padded jika missing
    'frequency': float,
    'signal_strength': float,
    # ... fitur prekursor lainnya
}
```

### Edge Construction

#### 1. Adjacency Matrix Calculation
Untuk setiap pasangan stasiun (i, j), hitung jarak Haversine:

```python
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Menghitung jarak great-circle antara dua titik di bumi.
    
    Returns:
        distance in kilometers
    """
    R = 6371.0  # Radius bumi dalam km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance
```

#### 2. Tectonic Penalty
Berdasarkan region_idx dari setiap stasiun:

```python
def calculate_tectonic_penalty(region_i, region_j):
    """
    Penalti untuk edge yang melintasi batas tektonik.
    
    Args:
        region_i: Region index stasiun i (0=Sunda, 1=Wallacea, 2=Sahul)
        region_j: Region index stasiun j
    
    Returns:
        P_fault: 0.5 jika berbeda region, 0.0 jika sama
    """
    if region_i != region_j:
        return 0.5  # Penalti untuk cross-tectonic edge
    else:
        return 0.0  # No penalty untuk same-region edge
```

#### 3. Edge Attributes
Setiap edge memiliki atribut:

```python
edge_attr = {
    'distance': float,        # Jarak Haversine (km)
    'tectonic_penalty': float, # 0.0 atau 0.5
    'weight': float           # 1 / (distance + epsilon)
}
```

### Graph Connectivity Strategy

**Option 1: Fully Connected Graph**
- Setiap stasiun terhubung ke semua stasiun lainnya
- Total edges: 24 × 23 / 2 = 276 edges (undirected)
- Pro: Tidak ada informasi yang hilang
- Con: Komputasi lebih berat

**Option 2: K-Nearest Neighbors (KNN)**
- Setiap stasiun hanya terhubung ke K tetangga terdekat
- K = 5-8 (recommended)
- Pro: Lebih efisien, fokus pada neighbor yang relevan
- Con: Mungkin kehilangan koneksi long-range

**Rekomendasi**: Mulai dengan Fully Connected, optimize ke KNN jika diperlukan.

## Prekursor Window & Label Assignment

### Jendela Prekursor
Untuk setiap gempa signifikan (Mw ≥ 5.0):
- **Prekursor Window**: 1-14 hari sebelum gempa
- **Background Noise**: Hari-hari tanpa gempa dalam window

### Label Extraction

Untuk setiap snapshot graf pada hari `t`:

```python
def extract_labels(date_t, earthquake_catalog, precursor_days=14):
    """
    Ekstrak 4 label untuk snapshot graf pada hari t.
    
    Returns:
        labels = {
            'event': 0 or 1,
            'magnitude': float or None,
            'azimuth': float or None,
            'distances': list[float] or None  # 24 distances (satu per stasiun)
        }
    """
    # Cari gempa dalam window [t, t+precursor_days]
    future_window = earthquake_catalog[
        (earthquake_catalog['time'] >= date_t) &
        (earthquake_catalog['time'] <= date_t + timedelta(days=precursor_days)) &
        (earthquake_catalog['mag'] >= 5.0)
    ]
    
    if len(future_window) == 0:
        # Background noise
        return {
            'event': 0,
            'magnitude': None,
            'azimuth': None,
            'distances': None
        }
    
    # Ambil gempa terdekat (earliest)
    earthquake = future_window.iloc[0]
    
    # Hitung azimuth dari centroid stasiun ke episenter
    centroid_lat = stations['latitude'].mean()
    centroid_lon = stations['longitude'].mean()
    azimuth = calculate_azimuth(
        centroid_lat, centroid_lon,
        earthquake['latitude'], earthquake['longitude']
    )
    
    # Hitung jarak dari setiap stasiun ke episenter
    distances = []
    for _, station in stations.iterrows():
        dist = haversine_distance(
            station['latitude'], station['longitude'],
            earthquake['latitude'], earthquake['longitude']
        )
        distances.append(dist)
    
    return {
        'event': 1,
        'magnitude': earthquake['mag'],
        'azimuth': azimuth,
        'distances': distances
    }
```

### Azimuth Calculation

```python
def calculate_azimuth(lat1, lon1, lat2, lon2):
    """
    Menghitung azimuth (bearing) dari titik 1 ke titik 2.
    
    Returns:
        azimuth in degrees (0-360)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
    
    azimuth_rad = math.atan2(x, y)
    azimuth_deg = math.degrees(azimuth_rad)
    
    # Normalize to 0-360
    azimuth_deg = (azimuth_deg + 360) % 360
    
    return azimuth_deg
```

## Handling Missing Data

### Strategi Zero-Padding & Masking

Jika stasiun tidak memiliki data pada hari tertentu:

```python
# Node features
node_features['is_active'] = False
node_features['amplitude'] = 0.0
node_features['frequency'] = 0.0
# ... semua fitur seismik = 0.0

# Masking tensor (untuk GAT)
node_mask = torch.tensor([1 if active else 0 for active in is_active_list])
```

GAT akan menggunakan mask ini untuk mengabaikan node yang tidak aktif dalam aggregation.

## Output Format

### PyTorch Geometric Format

```python
from torch_geometric.data import Data

graph_snapshot = Data(
    x=node_features,              # [24, num_features]
    edge_index=edge_index,        # [2, num_edges]
    edge_attr=edge_attr,          # [num_edges, edge_features]
    y_event=label_event,          # [1]
    y_mag=label_magnitude,        # [1]
    y_azm=label_azimuth,          # [1]
    y_dist=label_distances,       # [24]
    node_mask=node_mask,          # [24]
    date=date_t,                  # datetime
    space_weather=space_weather   # dict: {Kp, Dst, ...}
)
```

### HDF5 Hierarchical Format (Alternative)

```
dataset_graphs.h5
├── train/
│   ├── 2018-01-01/
│   │   ├── node_features [24, F]
│   │   ├── edge_index [2, E]
│   │   ├── edge_attr [E, 3]
│   │   ├── labels {event, mag, azm, dist}
│   │   └── metadata {date, Kp, Dst}
│   ├── 2018-01-02/
│   └── ...
├── val/
└── test/
```

## Implementation Steps

1. **Load Clean Data**
   - lokasi_stasiun_clean.csv
   - earthquake_catalog_clean.csv

2. **Initialize Station Graph**
   - Assign region_idx to each station
   - Calculate adjacency matrix
   - Calculate tectonic penalties

3. **Iterate Over Days**
   - For each day from 2018-01-01 to 2026-12-31:
     - Create node features (with zero-padding if needed)
     - Extract labels from earthquake catalog
     - Create graph snapshot
     - Save to appropriate split (train/val/test)

4. **Save Dataset**
   - PyTorch Geometric: `dataset_graphs.pt`
   - HDF5: `dataset_graphs.h5`

## Validation Checks

- [ ] Total snapshots = jumlah hari dalam range
- [ ] Setiap snapshot memiliki 24 nodes
- [ ] Edge count konsisten (276 untuk fully connected)
- [ ] Label event=1 hanya untuk hari dalam prekursor window
- [ ] Distances selalu positif dan realistis (< 5000 km)
- [ ] Azimuth dalam range 0-360°

## Next Steps

Setelah graph construction selesai:
1. Visualisasi beberapa snapshot graf
2. Statistik distribusi labels
3. Lanjut ke **Tahap 3: Physics-Informed Loss Design**

## Changelog

- **2026-05-02**: Dokumentasi design dibuat
- **Pending**: Implementasi script setelah data cleaning selesai
