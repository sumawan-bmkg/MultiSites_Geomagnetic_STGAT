"""
ANTIGRAVITY Project - Dataset Fix & Transformation
===================================================

Memperbaiki data leakage, membersihkan anomali magnitudo, dan
mentransformasi dataset dari single-station ke multi-station graph format.

Transformasi:
- Input: scalogram_v8_true_negatives.h5 (per-station)
- Output: scalogram_v9_multistation_graph.h5 (multi-station graph)

Author: ANTIGRAVITY Team
Date: 2026-05-02
"""

import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import os
import sys

# Configuration
INPUT_FILE = 'scalogram_v8_true_negatives.h5'
OUTPUT_FILE = 'scalogram_v9_multistation_graph.h5'

# Split dates
TRAIN_END_DATE = '2023-12-31'
VAL_START_DATE = '2024-01-01'
VAL_END_DATE = '2025-03-31'
TEST_START_DATE = '2026-01-01'

# Magnitude threshold
MIN_MAGNITUDE = 4.0

# Station list (will be extracted from data)
STATION_LIST = []


def extract_metadata_from_filename(filename):
    """
    Extract station code and date from filename.
    
    Example: 'event_ALR_20250824.npy' -> ('ALR', '2025-08-24')
    """
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8')
    
    parts = filename.split('_')
    if len(parts) >= 3:
        station = parts[1]
        date_str = parts[2].replace('.npy', '')
        
        # Parse YYYYMMDD
        if len(date_str) == 8:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            date = pd.Timestamp(year=year, month=month, day=day)
            return station, date
    
    return None, None


def load_and_merge_data():
    """
    TAHAP 1: Load dan merge semua data dari train dan val.
    
    Returns:
        DataFrame dengan kolom: date, station, tensor_idx, event, mag, azm, cosmic
    """
    print("\n" + "="*70)
    print("TAHAP 1: EKSTRAKSI & MERGE DATA")
    print("="*70)
    
    all_data = []
    
    with h5py.File(INPUT_FILE, 'r') as f:
        # Process train group
        print("\nProcessing train group...")
        if 'train' in f:
            train_group = f['train']
            n_samples = len(train_group['meta'])
            
            for i in tqdm(range(n_samples), desc="Loading train"):
                meta = train_group['meta'][i]
                station, date = extract_metadata_from_filename(meta)
                
                if station and date:
                    all_data.append({
                        'date': date,
                        'station': station,
                        'tensor_idx': i,
                        'group': 'train',
                        'event': int(train_group['label_event'][i]),
                        'mag': float(train_group['label_mag'][i]),
                        'azm': float(train_group['label_azm'][i]),
                        'cosmic_kp': float(train_group['cosmic_features'][i, 0]),
                        'cosmic_dst': float(train_group['cosmic_features'][i, 1])
                    })
        
        # Process val group
        print("\nProcessing val group...")
        if 'val' in f:
            val_group = f['val']
            n_samples = len(val_group['meta'])
            
            for i in tqdm(range(n_samples), desc="Loading val"):
                meta = val_group['meta'][i]
                station, date = extract_metadata_from_filename(meta)
                
                if station and date:
                    all_data.append({
                        'date': date,
                        'station': station,
                        'tensor_idx': i,
                        'group': 'val',
                        'event': int(val_group['label_event'][i]),
                        'mag': float(val_group['label_mag'][i]),
                        'azm': float(val_group['label_azm'][i]),
                        'cosmic_kp': float(val_group['cosmic_features'][i, 0]),
                        'cosmic_dst': float(val_group['cosmic_features'][i, 1])
                    })
    
    df = pd.DataFrame(all_data)
    
    print(f"\n✓ Loaded {len(df)} total samples")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Unique stations: {df['station'].nunique()}")
    print(f"  Unique dates: {df['date'].nunique()}")
    
    return df


def chronological_resplit(df):
    """
    TAHAP 1: Re-split data secara kronologis.
    
    Returns:
        train_df, val_df, test_df
    """
    print("\n" + "="*70)
    print("TAHAP 1: RE-SPLIT KRONOLOGIS")
    print("="*70)
    
    train_end = pd.to_datetime(TRAIN_END_DATE)
    val_start = pd.to_datetime(VAL_START_DATE)
    val_end = pd.to_datetime(VAL_END_DATE)
    test_start = pd.to_datetime(TEST_START_DATE)
    
    # Split by date
    train_df = df[df['date'] <= train_end].copy()
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()
    test_df = df[df['date'] >= test_start].copy()
    
    print(f"\n✓ Chronological split completed:")
    print(f"  Train: {len(train_df)} samples ({train_df['date'].min().date()} to {train_df['date'].max().date() if len(train_df) > 0 else 'N/A'})")
    print(f"  Val:   {len(val_df)} samples ({val_df['date'].min().date() if len(val_df) > 0 else 'N/A'} to {val_df['date'].max().date() if len(val_df) > 0 else 'N/A'})")
    print(f"  Test:  {len(test_df)} samples ({test_df['date'].min().date() if len(test_df) > 0 else 'N/A'} to {test_df['date'].max().date() if len(test_df) > 0 else 'N/A'})")
    
    # Verify no overlap
    if len(train_df) > 0 and len(val_df) > 0:
        assert train_df['date'].max() < val_df['date'].min(), "Train and Val overlap!"
    if len(val_df) > 0 and len(test_df) > 0:
        assert val_df['date'].max() < test_df['date'].min(), "Val and Test overlap!"
    
    print("  ✓ No temporal overlap detected")
    
    return train_df, val_df, test_df


def clean_magnitude_anomalies(df):
    """
    TAHAP 2: Bersihkan anomali magnitudo.
    
    Rules:
    - If event=1 and mag=0 or mag<4.0: REMOVE
    - If event=0: set mag=0.0 (dummy value)
    
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*70)
    print("TAHAP 2: PEMBERSIHAN MAGNITUDO")
    print("="*70)
    
    initial_count = len(df)
    
    # Find anomalies: event=1 but invalid magnitude
    anomalies = df[(df['event'] == 1) & ((df['mag'] == 0.0) | (df['mag'] < MIN_MAGNITUDE))]
    
    print(f"\nFound {len(anomalies)} anomalies (event=1 but mag=0 or mag<{MIN_MAGNITUDE})")
    
    if len(anomalies) > 0:
        print("\nSample anomalies:")
        print(anomalies[['date', 'station', 'event', 'mag']].head(10))
    
    # Remove anomalies
    df_clean = df[~((df['event'] == 1) & ((df['mag'] == 0.0) | (df['mag'] < MIN_MAGNITUDE)))].copy()
    
    # Set mag=0.0 for background noise (event=0)
    df_clean.loc[df_clean['event'] == 0, 'mag'] = 0.0
    
    removed_count = initial_count - len(df_clean)
    
    print(f"\n✓ Removed {removed_count} anomalous samples")
    print(f"✓ Remaining: {len(df_clean)} samples")
    print(f"✓ Background noise (event=0) magnitudes set to 0.0")
    
    return df_clean


def create_multistation_graphs(df, split_name):
    """
    TAHAP 3: Transform ke multi-station graph format.
    
    For each unique date:
    - Create tensor of shape (num_stations, 3, 128, 1440)
    - Fill with station data (zero-padding for missing stations)
    - Aggregate labels
    
    Returns:
        dict with keys: tensors, dates, events, mags, azms, dists, cosmic
    """
    print(f"\n" + "="*70)
    print(f"TAHAP 3: TRANSFORMASI MULTI-STATION GRAPH ({split_name})")
    print("="*70)
    
    # Get unique stations and dates
    unique_stations = sorted(df['station'].unique())
    unique_dates = sorted(df['date'].unique())
    
    global STATION_LIST
    if len(STATION_LIST) == 0:
        STATION_LIST = unique_stations
    
    num_stations = len(STATION_LIST)
    num_dates = len(unique_dates)
    
    print(f"\n✓ Stations: {num_stations} ({STATION_LIST})")
    print(f"✓ Unique dates: {num_dates}")
    
    # Create station to index mapping
    station_to_idx = {station: idx for idx, station in enumerate(STATION_LIST)}
    
    # Initialize arrays
    # Note: We'll load tensors on-the-fly from HDF5
    graph_tensors = []
    graph_dates = []
    graph_events = []
    graph_mags = []
    graph_azms = []
    graph_cosmic = []
    
    # Open HDF5 file for reading tensors
    with h5py.File(INPUT_FILE, 'r') as f:
        # Group by date
        for date in tqdm(unique_dates, desc=f"Creating {split_name} graphs"):
            date_data = df[df['date'] == date]
            
            # Initialize multi-station tensor (zero-padded)
            multi_tensor = np.zeros((num_stations, 3, 128, 1440), dtype=np.float16)
            azm_array = np.zeros(num_stations, dtype=np.float32)
            
            # Track which stations are active
            active_stations = []
            
            # Aggregate data for this date
            event_label = 0
            mag_label = 0.0
            cosmic_kp = 0.0
            cosmic_dst = 0.0
            
            for _, row in date_data.iterrows():
                station = row['station']
                if station not in station_to_idx:
                    continue
                
                station_idx = station_to_idx[station]
                active_stations.append(station_idx)
                
                # Load tensor from original HDF5
                group_name = row['group']
                tensor_idx = row['tensor_idx']
                
                tensor = f[group_name]['tensors'][tensor_idx]
                multi_tensor[station_idx] = tensor
                
                # Store azimuth for this station
                azm_array[station_idx] = row['azm']
                
                # Aggregate labels (take max event, max mag)
                if row['event'] > event_label:
                    event_label = row['event']
                if row['mag'] > mag_label:
                    mag_label = row['mag']
                
                # Take first cosmic features (should be same for all stations on same date)
                if cosmic_kp == 0.0:
                    cosmic_kp = row['cosmic_kp']
                    cosmic_dst = row['cosmic_dst']
            
            # Store graph
            graph_tensors.append(multi_tensor)
            graph_dates.append(date)
            graph_events.append(event_label)
            graph_mags.append(mag_label)
            graph_azms.append(azm_array)
            graph_cosmic.append([cosmic_kp, cosmic_dst])
    
    print(f"\n✓ Created {len(graph_tensors)} multi-station graph snapshots")
    print(f"  Shape per graph: ({num_stations}, 3, 128, 1440)")
    print(f"  Event samples: {sum(graph_events)}")
    print(f"  Background samples: {len(graph_events) - sum(graph_events)}")
    
    return {
        'tensors': np.array(graph_tensors),
        'dates': graph_dates,
        'events': np.array(graph_events, dtype=np.int8),
        'mags': np.array(graph_mags, dtype=np.float32),
        'azms': np.array(graph_azms, dtype=np.float32),
        'cosmic': np.array(graph_cosmic, dtype=np.float32)
    }


def save_multistation_dataset(train_data, val_data, test_data):
    """
    TAHAP 4: Simpan dataset baru ke HDF5.
    """
    print("\n" + "="*70)
    print("TAHAP 4: SIMPAN DATASET BARU")
    print("="*70)
    
    print(f"\nCreating {OUTPUT_FILE}...")
    
    with h5py.File(OUTPUT_FILE, 'w') as f:
        # Save train data
        if train_data and len(train_data['tensors']) > 0:
            print("\nSaving train group...")
            train_group = f.create_group('train')
            
            train_group.create_dataset('tensors', data=train_data['tensors'], 
                                      compression='gzip', compression_opts=4)
            train_group.create_dataset('dates', data=[str(d) for d in train_data['dates']])
            train_group.create_dataset('label_event', data=train_data['events'])
            train_group.create_dataset('label_mag', data=train_data['mags'])
            train_group.create_dataset('label_azm', data=train_data['azms'])
            train_group.create_dataset('cosmic_features', data=train_data['cosmic'])
            
            # Save station list as attribute
            train_group.attrs['stations'] = STATION_LIST
            train_group.attrs['num_stations'] = len(STATION_LIST)
            
            print(f"  ✓ Train: {len(train_data['tensors'])} graphs")
            print(f"     Shape: {train_data['tensors'].shape}")
        
        # Save val data
        if val_data and len(val_data['tensors']) > 0:
            print("\nSaving val group...")
            val_group = f.create_group('val')
            
            val_group.create_dataset('tensors', data=val_data['tensors'],
                                    compression='gzip', compression_opts=4)
            val_group.create_dataset('dates', data=[str(d) for d in val_data['dates']])
            val_group.create_dataset('label_event', data=val_data['events'])
            val_group.create_dataset('label_mag', data=val_data['mags'])
            val_group.create_dataset('label_azm', data=val_data['azms'])
            val_group.create_dataset('cosmic_features', data=val_data['cosmic'])
            
            val_group.attrs['stations'] = STATION_LIST
            val_group.attrs['num_stations'] = len(STATION_LIST)
            
            print(f"  ✓ Val: {len(val_data['tensors'])} graphs")
            print(f"     Shape: {val_data['tensors'].shape}")
        
        # Save test data
        if test_data and len(test_data['tensors']) > 0:
            print("\nSaving test group...")
            test_group = f.create_group('test')
            
            test_group.create_dataset('tensors', data=test_data['tensors'],
                                     compression='gzip', compression_opts=4)
            test_group.create_dataset('dates', data=[str(d) for d in test_data['dates']])
            test_group.create_dataset('label_event', data=test_data['events'])
            test_group.create_dataset('label_mag', data=test_data['mags'])
            test_group.create_dataset('label_azm', data=test_data['azms'])
            test_group.create_dataset('cosmic_features', data=test_data['cosmic'])
            
            test_group.attrs['stations'] = STATION_LIST
            test_group.attrs['num_stations'] = len(STATION_LIST)
            
            print(f"  ✓ Test: {len(test_data['tensors'])} graphs")
            print(f"     Shape: {test_data['tensors'].shape}")
        
        # Save global metadata
        f.attrs['format'] = 'multi-station-graph'
        f.attrs['version'] = 'v9'
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['num_stations'] = len(STATION_LIST)
        f.attrs['stations'] = STATION_LIST
        f.attrs['train_end_date'] = TRAIN_END_DATE
        f.attrs['val_start_date'] = VAL_START_DATE
        f.attrs['val_end_date'] = VAL_END_DATE
        f.attrs['test_start_date'] = TEST_START_DATE
        f.attrs['min_magnitude'] = MIN_MAGNITUDE
    
    print(f"\n✓ Dataset saved to {OUTPUT_FILE}")
    
    # Print file size
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")


def main():
    """Main execution."""
    print("="*70)
    print("ANTIGRAVITY - DATASET FIX & TRANSFORMATION")
    print("="*70)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Started: {datetime.now()}")
    print()
    
    # Check input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ERROR: Input file not found: {INPUT_FILE}")
        return 1
    
    try:
        # TAHAP 1: Load and merge data
        df = load_and_merge_data()
        
        # TAHAP 2: Clean magnitude anomalies
        df_clean = clean_magnitude_anomalies(df)
        
        # TAHAP 1 (continued): Re-split chronologically
        train_df, val_df, test_df = chronological_resplit(df_clean)
        
        # TAHAP 3: Transform to multi-station graphs
        train_data = None
        val_data = None
        test_data = None
        
        if len(train_df) > 0:
            train_data = create_multistation_graphs(train_df, 'train')
        else:
            print("\n⚠️  WARNING: No train data after split")
        
        if len(val_df) > 0:
            val_data = create_multistation_graphs(val_df, 'val')
        else:
            print("\n⚠️  WARNING: No val data after split")
        
        if len(test_df) > 0:
            test_data = create_multistation_graphs(test_df, 'test')
        else:
            print("\n⚠️  WARNING: No test data after split")
        
        # TAHAP 4: Save new dataset
        save_multistation_dataset(train_data, val_data, test_data)
        
        # Summary
        print("\n" + "="*70)
        print("TRANSFORMATION COMPLETE!")
        print("="*70)
        print(f"\n✅ Successfully created multi-station graph dataset")
        print(f"✅ Fixed data leakage with chronological split")
        print(f"✅ Cleaned magnitude anomalies")
        print(f"✅ Transformed to multi-station format")
        print(f"\nOutput file: {OUTPUT_FILE}")
        print(f"Format: (num_days, {len(STATION_LIST)}, 3, 128, 1440)")
        print(f"Stations: {len(STATION_LIST)}")
        
        print(f"\nCompleted: {datetime.now()}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
