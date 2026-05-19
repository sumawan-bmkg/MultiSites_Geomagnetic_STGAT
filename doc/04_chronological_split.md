# Tahap 4: Strict Chronological Split

**Tanggal**: 2 Mei 2026  
**Status**: Design Phase

## Tujuan

Membagi dataset graph snapshots ke dalam 3 subset (train, validation, test) secara **ketat berdasarkan waktu global** untuk menghindari spatio-temporal data leakage.

## Prinsip Utama: Menghindari Data Leakage

### Apa itu Spatio-Temporal Data Leakage?

**Data leakage** terjadi ketika informasi dari masa depan "bocor" ke dalam training set, menyebabkan model terlihat bagus di validation tetapi gagal di real-world deployment.

### Contoh Data Leakage (SALAH ❌)

```python
# SALAH: Random split tanpa memperhatikan waktu
all_graphs = load_all_graphs()
random.shuffle(all_graphs)

train = all_graphs[:7000]  # Bisa berisi data dari 2024, 2025, 2026
val = all_graphs[7000:8500]  # Bisa berisi data dari 2018, 2019
test = all_graphs[8500:]  # Mixed dates
```

**Masalah**:
- Model bisa "melihat" pola dari 2024 saat training, lalu di-test dengan data 2023
- Tidak realistis: Di real-world, kita tidak bisa melihat masa depan

### Chronological Split yang Benar (BENAR ✅)

```python
# BENAR: Split berdasarkan tanggal global
train_graphs = graphs[graphs['date'] <= '2023-12-31']
val_graphs = graphs[(graphs['date'] >= '2024-01-01') & 
                    (graphs['date'] <= '2025-03-31')]
test_graphs = graphs[graphs['date'] >= '2026-01-01']
```

**Keuntungan**:
- Model hanya belajar dari masa lalu
- Validation dan test benar-benar "unseen" dalam konteks temporal
- Realistis untuk deployment

## Pembagian Dataset

### Timeline

```
|-------- TRAIN --------|--- VAL ---|---- TEST ----|
2018                    2024       2025           2026+
```

### Train Set
- **Periode**: Awal observasi - 31 Desember 2023
- **Tujuan**: Pembelajaran pola prekursor gempa
- **Expected Size**: ~2,190 hari (6 tahun × 365 hari)
- **Karakteristik**:
  - Mencakup berbagai kondisi seismik
  - Multiple significant earthquakes (Mw ≥ 5.0)
  - Berbagai kondisi space weather

### Validation Set
- **Periode**: 1 Januari 2024 - 31 Maret 2025
- **Tujuan**: 
  - Hyperparameter tuning
  - Early stopping
  - Evaluasi Cosmic Gating module
- **Expected Size**: ~455 hari (15 bulan)
- **Karakteristik Khusus**:
  - **Mei 2024**: Badai matahari ekstrem (Kp = 9.0)
  - Menguji robustness model terhadap anomali space weather

### Test Set (Blind Test)
- **Periode**: 1 Januari 2026 - seterusnya
- **Tujuan**: Evaluasi final pada data yang benar-benar unseen
- **Expected Size**: ~120+ hari (4+ bulan hingga Mei 2026)
- **Karakteristik**:
  - Completely blind: Model tidak pernah "melihat" data ini
  - Simulasi real-world deployment

## Implementation

### Step 1: Load All Graph Snapshots

```python
import torch
from torch_geometric.data import Data
import pandas as pd
from datetime import datetime

def load_all_graphs(graph_file='data/processed/dataset_graphs.pt'):
    """
    Load semua graph snapshots dari file.
    
    Returns:
        list of Data objects, each with 'date' attribute
    """
    graphs = torch.load(graph_file)
    print(f"Loaded {len(graphs)} graph snapshots")
    return graphs
```

### Step 2: Chronological Split Function

```python
def chronological_split(graphs, 
                        train_end='2023-12-31',
                        val_end='2025-03-31',
                        test_start='2026-01-01'):
    """
    Split graphs berdasarkan tanggal global.
    
    Args:
        graphs: list of Data objects with 'date' attribute
        train_end: Last date for training set
        val_end: Last date for validation set
        test_start: First date for test set
    
    Returns:
        train_graphs, val_graphs, test_graphs
    """
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    test_start_dt = pd.to_datetime(test_start)
    
    train_graphs = []
    val_graphs = []
    test_graphs = []
    
    for graph in graphs:
        graph_date = pd.to_datetime(graph.date)
        
        if graph_date <= train_end_dt:
            train_graphs.append(graph)
        elif graph_date <= val_end_dt:
            val_graphs.append(graph)
        elif graph_date >= test_start_dt:
            test_graphs.append(graph)
        # Dates between val_end and test_start are discarded (gap period)
    
    print(f"Train: {len(train_graphs)} graphs")
    print(f"Val: {len(val_graphs)} graphs")
    print(f"Test: {len(test_graphs)} graphs")
    
    return train_graphs, val_graphs, test_graphs
```

### Step 3: Validate Split

```python
def validate_split(train_graphs, val_graphs, test_graphs):
    """
    Validasi bahwa split benar-benar chronological dan tidak ada overlap.
    """
    # Extract dates
    train_dates = [pd.to_datetime(g.date) for g in train_graphs]
    val_dates = [pd.to_datetime(g.date) for g in val_graphs]
    test_dates = [pd.to_datetime(g.date) for g in test_graphs]
    
    # Check 1: No overlap
    train_max = max(train_dates)
    val_min = min(val_dates)
    val_max = max(val_dates)
    test_min = min(test_dates)
    
    assert train_max < val_min, "Train and Val overlap!"
    assert val_max < test_min, "Val and Test overlap!"
    
    print("✓ No temporal overlap between splits")
    
    # Check 2: Chronological order within each split
    assert train_dates == sorted(train_dates), "Train not chronological!"
    assert val_dates == sorted(val_dates), "Val not chronological!"
    assert test_dates == sorted(test_dates), "Test not chronological!"
    
    print("✓ Each split is chronologically ordered")
    
    # Check 3: Date ranges
    print(f"\nTrain: {min(train_dates).date()} to {max(train_dates).date()}")
    print(f"Val:   {min(val_dates).date()} to {max(val_dates).date()}")
    print(f"Test:  {min(test_dates).date()} to {max(test_dates).date()}")
    
    return True
```

### Step 4: Save Split Datasets

```python
def save_split_datasets(train_graphs, val_graphs, test_graphs,
                        output_dir='data'):
    """
    Save split datasets ke file terpisah.
    """
    import os
    
    # Save as PyTorch files
    torch.save(train_graphs, os.path.join(output_dir, 'train', 'graphs.pt'))
    torch.save(val_graphs, os.path.join(output_dir, 'val', 'graphs.pt'))
    torch.save(test_graphs, os.path.join(output_dir, 'test', 'graphs.pt'))
    
    print(f"✓ Saved train set: {len(train_graphs)} graphs")
    print(f"✓ Saved val set: {len(val_graphs)} graphs")
    print(f"✓ Saved test set: {len(test_graphs)} graphs")
    
    # Save metadata
    metadata = {
        'train': {
            'count': len(train_graphs),
            'date_range': (
                str(min([g.date for g in train_graphs])),
                str(max([g.date for g in train_graphs]))
            )
        },
        'val': {
            'count': len(val_graphs),
            'date_range': (
                str(min([g.date for g in val_graphs])),
                str(max([g.date for g in val_graphs]))
            )
        },
        'test': {
            'count': len(test_graphs),
            'date_range': (
                str(min([g.date for g in test_graphs])),
                str(max([g.date for g in test_graphs]))
            )
        }
    }
    
    import json
    with open(os.path.join(output_dir, 'split_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Saved split metadata")
```

## Class Distribution Analysis

### Importance
Setelah split, penting untuk menganalisis distribusi class (event vs background) di setiap subset:

```python
def analyze_class_distribution(graphs, split_name='Train'):
    """
    Analisis distribusi event vs background noise.
    """
    event_count = sum([1 for g in graphs if g.y_event.item() == 1])
    background_count = len(graphs) - event_count
    
    event_ratio = event_count / len(graphs)
    
    print(f"\n{split_name} Set Class Distribution:")
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Event (prekursor): {event_count} ({event_ratio*100:.2f}%)")
    print(f"  Background: {background_count} ({(1-event_ratio)*100:.2f}%)")
    print(f"  Imbalance ratio: 1:{background_count/max(event_count, 1):.1f}")
    
    return event_ratio
```

**Expected Imbalance**:
- Event (prekursor): ~5-15% dari total
- Background: ~85-95% dari total
- Imbalance ratio: 1:6 hingga 1:20

**Handling Strategy**:
- Weighted loss function (sudah diimplementasi di Tahap 3)
- Focal loss untuk event detection
- Oversampling event graphs (optional)

## Magnitude Distribution Analysis

```python
def analyze_magnitude_distribution(graphs, split_name='Train'):
    """
    Analisis distribusi magnitude gempa.
    """
    magnitudes = [g.y_mag.item() for g in graphs if g.y_event.item() == 1]
    
    if len(magnitudes) == 0:
        print(f"{split_name}: No events found")
        return
    
    import numpy as np
    print(f"\n{split_name} Set Magnitude Distribution:")
    print(f"  Count: {len(magnitudes)}")
    print(f"  Mean: {np.mean(magnitudes):.2f}")
    print(f"  Std: {np.std(magnitudes):.2f}")
    print(f"  Min: {np.min(magnitudes):.2f}")
    print(f"  Max: {np.max(magnitudes):.2f}")
    print(f"  Median: {np.median(magnitudes):.2f}")
    
    # Histogram
    bins = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 10.0]
    hist, _ = np.histogram(magnitudes, bins=bins)
    
    print("\n  Magnitude Bins:")
    for i in range(len(bins)-1):
        print(f"    {bins[i]:.1f} - {bins[i+1]:.1f}: {hist[i]} events")
```

## Space Weather Analysis (Validation Set)

Khusus untuk validation set, analisis space weather events:

```python
def analyze_space_weather(val_graphs):
    """
    Analisis space weather events di validation set.
    Fokus pada Mei 2024 (Kp = 9.0 storm).
    """
    import pandas as pd
    
    # Extract space weather data
    dates = [pd.to_datetime(g.date) for g in val_graphs]
    kp_values = [g.space_weather.get('Kp', 0) for g in val_graphs]
    
    # Find extreme events
    extreme_kp = [(d, kp) for d, kp in zip(dates, kp_values) if kp >= 7.0]
    
    print("\nSpace Weather Analysis (Validation Set):")
    print(f"  Total days: {len(val_graphs)}")
    print(f"  Days with Kp ≥ 7: {len(extreme_kp)}")
    
    if extreme_kp:
        print("\n  Extreme Space Weather Events:")
        for date, kp in extreme_kp:
            print(f"    {date.date()}: Kp = {kp}")
    
    # Check for May 2024 storm
    may_2024 = [g for g in val_graphs 
                if pd.to_datetime(g.date).year == 2024 
                and pd.to_datetime(g.date).month == 5]
    
    if may_2024:
        max_kp_may = max([g.space_weather.get('Kp', 0) for g in may_2024])
        print(f"\n  May 2024 Maximum Kp: {max_kp_may}")
        if max_kp_may >= 9.0:
            print("  ✓ Extreme storm (Kp=9.0) detected in validation set")
```

## Complete Split Pipeline

```python
def main():
    """
    Complete pipeline untuk chronological split.
    """
    print("="*60)
    print("ANTIGRAVITY: Chronological Split Pipeline")
    print("="*60)
    
    # Step 1: Load graphs
    print("\n[1/5] Loading graph snapshots...")
    graphs = load_all_graphs('data/processed/dataset_graphs.pt')
    
    # Step 2: Chronological split
    print("\n[2/5] Performing chronological split...")
    train_graphs, val_graphs, test_graphs = chronological_split(graphs)
    
    # Step 3: Validate split
    print("\n[3/5] Validating split...")
    validate_split(train_graphs, val_graphs, test_graphs)
    
    # Step 4: Analyze distributions
    print("\n[4/5] Analyzing distributions...")
    analyze_class_distribution(train_graphs, 'Train')
    analyze_class_distribution(val_graphs, 'Validation')
    analyze_class_distribution(test_graphs, 'Test')
    
    analyze_magnitude_distribution(train_graphs, 'Train')
    analyze_magnitude_distribution(val_graphs, 'Validation')
    analyze_magnitude_distribution(test_graphs, 'Test')
    
    analyze_space_weather(val_graphs)
    
    # Step 5: Save split datasets
    print("\n[5/5] Saving split datasets...")
    save_split_datasets(train_graphs, val_graphs, test_graphs)
    
    print("\n" + "="*60)
    print("Chronological split completed successfully!")
    print("="*60)

if __name__ == '__main__':
    main()
```

## Validation Checklist

Sebelum melanjutkan ke training, pastikan:

- [ ] Train set hanya berisi data hingga 31 Des 2023
- [ ] Val set berisi data 1 Jan 2024 - 31 Mar 2025
- [ ] Test set berisi data mulai 1 Jan 2026
- [ ] Tidak ada temporal overlap antar splits
- [ ] Setiap split terurut secara chronological
- [ ] Class distribution analyzed dan documented
- [ ] Magnitude distribution reasonable
- [ ] Space weather events (Kp=9.0) ada di val set

## Next Steps

Setelah chronological split selesai:
1. Create DataLoader untuk PyTorch training
2. Implement model architecture (GAT + DPINN)
3. Lanjut ke **Tahap 5: Model Training**

## Changelog

- **2026-05-02**: Dokumentasi chronological split design dibuat
- **Pending**: Implementasi setelah graph construction selesai
