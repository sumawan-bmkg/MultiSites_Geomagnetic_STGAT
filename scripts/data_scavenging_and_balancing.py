"""
Data Scavenging & Balancing Script for ANTIGRAVITY Project
Tujuan: Kumpulkan semua data histori, perbaiki Dst, balance classes, dan buat multi-station graph
"""

import h5py
import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths to scan (exclude blindtest)
HDF5_FILES = [
    'scalogram_v8_true_negatives.h5',
    'scalogram_v8_hard_negatives.h5',
    'scalogram_v3_cosmic_final.h5',
]

# Earthquake catalog
EARTHQUAKE_CATALOG = 'intial/earthquake_catalog_2018_2025_merged_robust.csv'
STATION_LIST = 'intial/lokasi_stasiun.csv'
DST_FILE = 'intial/dst.txt'

# Output file
OUTPUT_FILE = 'dataset_v10_train_val_graphs.h5'

# Date ranges
TRAIN_END_DATE = '2023-12-31'
VAL_START_DATE = '2024-01-01'
VAL_END_DATE = '2025-03-31'
BLINDTEST_START_DATE = '2026-01-01'

# Balancing ratio (Event:Noise)
TARGET_RATIO = 4  # 1 Event : 4 Noise

# Minimum magnitude
MIN_MAGNITUDE = 4.0


# ============================================================================
# TAHAP 1: DEEP SEARCH & INVENTORY
# ============================================================================

def scan_hdf5_files():
    """Scan all HDF5 files and collect metadata."""
    print("="*70)
    print("TAHAP 1: DEEP SEARCH & INVENTORY")
    print("="*70)
    
    master_data = []
    
    for hdf5_file in HDF5_FILES:
        if not os.path.exists(hdf5_file):
            print(f"\n⚠️ File not found: {hdf5_file}")
            continue
        
        print(f"\n📁 Scanning: {hdf5_file}")
        print(f"   Size: {os.path.getsize(hdf5_file) / (1024**2):.2f} MB")
        
        try:
            with h5py.File(hdf5_file, 'r') as f:
                # Scan all groups
                for group_name in f.keys():
                    if group_name in ['train', 'val', 'test']:
                        group = f[group_name]
                        
                        # Extract metadata
                        if 'meta' in group:
                            meta = group['meta'][:]
                        elif 'dates' in group:
                            meta = group['dates'][:]
                        else:
                            print(f"   ⚠️ No metadata found in {group_name}")
                            continue
                        
                        # Extract labels
                        events = group['label_event'][:] if 'label_event' in group else None
                        mags = group['label_mag'][:] if 'label_mag' in group else None
                        azms = group['label_azm'][:] if 'label_azm' in group else None
                        
                        # Extract cosmic features
                        cosmic = group['cosmic_features'][:] if 'cosmic_features' in group else None
                        
                        # Extract tensors info
                        tensors_shape = group['tensors'].shape if 'tensors' in group else None
                        
                        # Parse dates from metadata
                        dates = []
                        stations = []
                        for m in meta:
                            if isinstance(m, bytes):
                                m = m.decode('utf-8')
                            
                            # Extract date and station from filename
                            # Format: STATION_YYYYMMDD_*.npy or similar
                            parts = str(m).split('_')
                            if len(parts) >= 2:
                                station = parts[0]
                                date_str = parts[1][:8]  # YYYYMMDD
                                try:
                                    date = pd.to_datetime(date_str, format='%Y%m%d')
                                    dates.append(date)
                                    stations.append(station)
                                except:
                                    dates.append(None)
                                    stations.append(None)
                            else:
                                dates.append(None)
                                stations.append(None)
                        
                        # Create dataframe for this group
                        n_samples = len(meta)
                        for i in range(n_samples):
                            record = {
                                'source_file': hdf5_file,
                                'source_group': group_name,
                                'index': i,
                                'date': dates[i],
                                'station': stations[i],
                                'meta': meta[i],
                                'event': events[i] if events is not None else None,
                                'magnitude': mags[i] if mags is not None else None,
                                'azimuth': azms[i] if azms is not None else None,
                                'kp': cosmic[i, 0] if cosmic is not None else None,
                                'dst': cosmic[i, 1] if cosmic is not None else None,
                            }
                            master_data.append(record)
                        
                        print(f"   ✓ {group_name}: {n_samples} samples")
        
        except Exception as e:
            print(f"   ❌ Error reading {hdf5_file}: {e}")
    
    # Create master dataframe
    df = pd.DataFrame(master_data)
    
    # Filter out None dates
    df = df[df['date'].notna()].copy()
    
    # Filter out blindtest dates (>= 2026-01-01)
    blindtest_date = pd.to_datetime(BLINDTEST_START_DATE)
    df = df[df['date'] < blindtest_date].copy()
    
    print(f"\n✓ Total samples collected: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Unique stations: {df['station'].nunique()}")
    print(f"  Events: {df['event'].sum()} ({df['event'].sum()/len(df)*100:.1f}%)")
    
    return df


# ============================================================================
# TAHAP 2: SPACE WEATHER ANOMALY FIX
# ============================================================================

def fix_space_weather(df):
    """Fix Dst anomalies by cross-referencing with external data."""
    print("\n" + "="*70)
    print("TAHAP 2: SPACE WEATHER ANOMALY FIX")
    print("="*70)
    
    # Check for constant Dst values
    dst_values = df['dst'].dropna()
    if len(dst_values) > 0:
        dst_unique = dst_values.nunique()
        dst_mean = dst_values.mean()
        dst_std = dst_values.std()
        
        print(f"\n📊 Dst Statistics:")
        print(f"   Unique values: {dst_unique}")
        print(f"   Mean: {dst_mean:.2f}")
        print(f"   Std: {dst_std:.2f}")
        
        # Check if Dst is corrupted (constant or very low variance)
        if dst_unique == 1 or dst_std < 1.0:
            print(f"\n⚠️ WARNING: Dst appears corrupted (constant or low variance)")
            
            # Try to load external Dst data
            if os.path.exists(DST_FILE):
                print(f"   Attempting to load Dst from: {DST_FILE}")
                try:
                    # Load Dst file (format may vary)
                    dst_data = pd.read_csv(DST_FILE, sep=r'\s+', header=None, 
                                          names=['year', 'month', 'day', 'hour', 'dst'])
                    dst_data['date'] = pd.to_datetime(dst_data[['year', 'month', 'day']])
                    
                    # Group by date and take daily average
                    dst_daily = dst_data.groupby('date')['dst'].mean().reset_index()
                    
                    # Merge with master dataframe
                    df = df.merge(dst_daily, on='date', how='left', suffixes=('_old', ''))
                    
                    # Fill missing Dst with old values
                    df['dst'] = df['dst'].fillna(df['dst_old'])
                    df = df.drop(columns=['dst_old'], errors='ignore')
                    
                    print(f"   ✓ Dst values updated from external file")
                    print(f"   New Dst range: {df['dst'].min():.2f} to {df['dst'].max():.2f}")
                
                except Exception as e:
                    print(f"   ❌ Error loading Dst file: {e}")
                    print(f"   🔴 CRITICAL: Dst data remains corrupted!")
            else:
                print(f"   ❌ Dst file not found: {DST_FILE}")
                print(f"   🔴 CRITICAL: Cannot fix Dst data!")
        else:
            print(f"   ✓ Dst data appears valid")
    
    # Check Kp values
    kp_values = df['kp'].dropna()
    if len(kp_values) > 0:
        print(f"\n📊 Kp Statistics:")
        print(f"   Range: {kp_values.min():.2f} to {kp_values.max():.2f}")
        print(f"   Mean: {kp_values.mean():.2f}")
        print(f"   ✓ Kp data appears valid")
    
    return df


# ============================================================================
# TAHAP 3: STRICT TEMPORAL SPLITTING
# ============================================================================

def temporal_split(df):
    """Split data chronologically into train and val sets."""
    print("\n" + "="*70)
    print("TAHAP 3: STRICT TEMPORAL SPLITTING")
    print("="*70)
    
    train_end = pd.to_datetime(TRAIN_END_DATE)
    val_start = pd.to_datetime(VAL_START_DATE)
    val_end = pd.to_datetime(VAL_END_DATE)
    
    # Split data
    df_train = df[df['date'] <= train_end].copy()
    df_val = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()
    
    print(f"\n📊 Train Set:")
    print(f"   Date range: {df_train['date'].min().date() if len(df_train) > 0 else 'N/A'} to "
          f"{df_train['date'].max().date() if len(df_train) > 0 else 'N/A'}")
    print(f"   Total samples: {len(df_train)}")
    print(f"   Events: {df_train['event'].sum() if len(df_train) > 0 else 0} "
          f"({df_train['event'].sum()/len(df_train)*100 if len(df_train) > 0 else 0:.1f}%)")
    print(f"   Noise: {len(df_train) - df_train['event'].sum() if len(df_train) > 0 else 0}")
    
    print(f"\n📊 Validation Set:")
    print(f"   Date range: {df_val['date'].min().date() if len(df_val) > 0 else 'N/A'} to "
          f"{df_val['date'].max().date() if len(df_val) > 0 else 'N/A'}")
    print(f"   Total samples: {len(df_val)}")
    print(f"   Events: {df_val['event'].sum() if len(df_val) > 0 else 0} "
          f"({df_val['event'].sum()/len(df_val)*100 if len(df_val) > 0 else 0:.1f}%)")
    print(f"   Noise: {len(df_val) - df_val['event'].sum() if len(df_val) > 0 else 0}")
    
    return df_train, df_val


# ============================================================================
# TAHAP 4: DATA BALANCING
# ============================================================================

def balance_dataset(df, set_name='train'):
    """Balance event and noise samples."""
    print(f"\n{'='*70}")
    print(f"TAHAP 4: DATA BALANCING ({set_name.upper()} SET)")
    print("="*70)
    
    if len(df) == 0:
        print(f"\n⚠️ No data in {set_name} set!")
        return df
    
    # Separate events and noise
    df_events = df[df['event'] == 1].copy()
    df_noise = df[df['event'] == 0].copy()
    
    print(f"\n📊 Before Balancing:")
    print(f"   Events: {len(df_events)}")
    print(f"   Noise: {len(df_noise)}")
    print(f"   Ratio: 1:{len(df_noise)/len(df_events) if len(df_events) > 0 else 0:.1f}")
    
    # Remove invalid events (mag == 0 or mag < MIN_MAGNITUDE)
    if len(df_events) > 0:
        invalid_events = df_events[(df_events['magnitude'] == 0) | 
                                   (df_events['magnitude'] < MIN_MAGNITUDE)]
        if len(invalid_events) > 0:
            print(f"\n⚠️ Removing {len(invalid_events)} invalid events (mag < {MIN_MAGNITUDE})")
            df_events = df_events[(df_events['magnitude'] > 0) & 
                                 (df_events['magnitude'] >= MIN_MAGNITUDE)].copy()
    
    # Preserve important events (Mw >= 5.0)
    if len(df_events) > 0:
        df_events_important = df_events[df_events['magnitude'] >= 5.0].copy()
        df_events_normal = df_events[df_events['magnitude'] < 5.0].copy()
        print(f"\n✓ Preserving {len(df_events_important)} important events (Mw >= 5.0)")
    else:
        df_events_important = pd.DataFrame()
        df_events_normal = pd.DataFrame()
    
    # Preserve May 2024 storm data (Kp >= 8.0)
    if set_name == 'val':
        may_2024_start = pd.to_datetime('2024-05-01')
        may_2024_end = pd.to_datetime('2024-05-31')
        df_may_storm = df[(df['date'] >= may_2024_start) & 
                         (df['date'] <= may_2024_end) & 
                         (df['kp'] >= 8.0)].copy()
        if len(df_may_storm) > 0:
            print(f"✓ Preserving {len(df_may_storm)} May 2024 storm samples (Kp >= 8.0)")
    else:
        df_may_storm = pd.DataFrame()
    
    # Calculate target noise count
    n_events = len(df_events)
    if n_events == 0:
        print(f"\n⚠️ WARNING: No events in {set_name} set!")
        print(f"   Keeping all noise samples")
        return df
    
    target_noise = n_events * TARGET_RATIO
    
    # Undersample noise if needed
    if len(df_noise) > target_noise:
        print(f"\n🔄 Undersampling noise: {len(df_noise)} → {int(target_noise)}")
        
        # Sample noise evenly across months
        df_noise['month'] = df_noise['date'].dt.to_period('M')
        noise_per_month = df_noise.groupby('month').size()
        
        # Calculate samples per month
        samples_per_month = int(target_noise / len(noise_per_month))
        
        sampled_noise = []
        for month in noise_per_month.index:
            month_data = df_noise[df_noise['month'] == month]
            n_sample = min(samples_per_month, len(month_data))
            sampled = month_data.sample(n=n_sample, random_state=42)
            sampled_noise.append(sampled)
        
        df_noise = pd.concat(sampled_noise, ignore_index=True)
        df_noise = df_noise.drop(columns=['month'])
    
    # Combine all data
    df_balanced = pd.concat([df_events, df_noise, df_may_storm], ignore_index=True)
    df_balanced = df_balanced.drop_duplicates(subset=['source_file', 'index'])
    
    print(f"\n📊 After Balancing:")
    print(f"   Events: {df_balanced['event'].sum()}")
    print(f"   Noise: {len(df_balanced) - df_balanced['event'].sum()}")
    print(f"   Ratio: 1:{(len(df_balanced) - df_balanced['event'].sum())/df_balanced['event'].sum():.1f}")
    print(f"   Total samples: {len(df_balanced)}")
    
    return df_balanced


# ============================================================================
# TAHAP 5: MULTI-STATION GRAPH TRANSFORMATION
# ============================================================================

def create_multistation_graphs(df, set_name='train'):
    """Transform per-station data to multi-station graph format."""
    print(f"\n{'='*70}")
    print(f"TAHAP 5: MULTI-STATION GRAPH TRANSFORMATION ({set_name.upper()})")
    print("="*70)
    
    if len(df) == 0:
        print(f"\n⚠️ No data to transform!")
        return None
    
    # Get unique stations
    stations = sorted(df['station'].unique())
    n_stations = len(stations)
    station_to_idx = {s: i for i, s in enumerate(stations)}
    
    print(f"\n📊 Stations found: {n_stations}")
    print(f"   {stations}")
    
    # Group by date
    grouped = df.groupby('date')
    dates = sorted(df['date'].unique())
    n_days = len(dates)
    
    print(f"\n📊 Creating graph snapshots:")
    print(f"   Total days: {n_days}")
    print(f"   Stations per day: {n_stations}")
    
    # Initialize lists for labels (lightweight)
    label_event_list = []
    label_mag_list = []
    label_azm_list = []
    cosmic_list = []
    dates_list = []
    
    # Load tensors from source files
    source_files = {}
    for source_file in df['source_file'].unique():
        if os.path.exists(source_file):
            source_files[source_file] = h5py.File(source_file, 'r')
    
    # Create output file directly to avoid memory issues
    temp_output = f'temp_{set_name}_graphs.h5'
    with h5py.File(temp_output, 'w') as out_f:
        # Create dataset for tensors
        tensors_ds = out_f.create_dataset('tensors', 
                                         shape=(n_days, n_stations, 3, 128, 1440),
                                         dtype=np.float16,
                                         compression='gzip',
                                         chunks=(1, n_stations, 3, 128, 1440))
        
        # Process each date
        for day_idx, date in enumerate(dates):
            if day_idx % 50 == 0:
                print(f"   Processing day {day_idx+1}/{n_days}...")
            
            day_data = grouped.get_group(date)
            
            # Create empty tensor for this day (n_stations, 3, 128, 1440)
            day_tensor = np.zeros((n_stations, 3, 128, 1440), dtype=np.float16)
            day_azm = np.zeros(n_stations, dtype=np.float32)
            
            # Fill tensor with available stations
            day_event = 0
            day_mag = 0.0
            day_kp = 0.0
            day_dst = 0.0
            
            for _, row in day_data.iterrows():
                station = row['station']
                if station not in station_to_idx:
                    continue
                
                station_idx = station_to_idx[station]
                
                # Load tensor from source file
                source_file = row['source_file']
                source_group = row['source_group']
                index = row['index']
                
                if source_file in source_files:
                    try:
                        tensor = source_files[source_file][source_group]['tensors'][index]
                        day_tensor[station_idx] = tensor
                        day_azm[station_idx] = row['azimuth'] if pd.notna(row['azimuth']) else 0.0
                    except Exception as e:
                        pass
                
                # Aggregate labels (take max event, max mag)
                if row['event'] == 1:
                    day_event = 1
                    if pd.notna(row['magnitude']) and row['magnitude'] > day_mag:
                        day_mag = row['magnitude']
                
                # Take average cosmic features
                if pd.notna(row['kp']):
                    day_kp = max(day_kp, row['kp'])
                if pd.notna(row['dst']):
                    day_dst = row['dst']  # Take last value
            
            # Write tensor to file
            tensors_ds[day_idx] = day_tensor
            
            # Append labels to lists
            label_event_list.append(day_event)
            label_mag_list.append(day_mag)
            label_azm_list.append(day_azm)
            cosmic_list.append([day_kp, day_dst])
            dates_list.append(date.strftime('%Y-%m-%d'))
            
            # Clear memory
            del day_tensor
    
    # Close source files
    for f in source_files.values():
        f.close()
    
    # Convert labels to arrays
    label_event = np.array(label_event_list, dtype=np.int8)
    label_mag = np.array(label_mag_list, dtype=np.float32)
    label_azm = np.array(label_azm_list, dtype=np.float32)
    cosmic_features = np.array(cosmic_list, dtype=np.float32)
    dates_array = np.array(dates_list, dtype='S10')
    
    print(f"\n✓ Graph snapshots created:")
    print(f"   Shape: ({n_days}, {n_stations}, 3, 128, 1440)")
    print(f"   Events: {label_event.sum()} ({label_event.sum()/len(label_event)*100:.1f}%)")
    print(f"   Magnitude range: {label_mag[label_mag > 0].min():.2f} - {label_mag.max():.2f}")
    
    return {
        'temp_file': temp_output,
        'label_event': label_event,
        'label_mag': label_mag,
        'label_azm': label_azm,
        'cosmic_features': cosmic_features,
        'dates': dates_array,
        'stations': stations,
    }


# ============================================================================
# TAHAP 6: FINAL OUTPUT
# ============================================================================

def save_dataset(train_data, val_data):
    """Save final dataset to HDF5."""
    print(f"\n{'='*70}")
    print("TAHAP 6: FINAL OUTPUT")
    print("="*70)
    
    print(f"\n💾 Saving to: {OUTPUT_FILE}")
    
    with h5py.File(OUTPUT_FILE, 'w') as f:
        # Save train set
        if train_data is not None:
            train_group = f.create_group('train')
            
            # Copy tensors from temp file
            with h5py.File(train_data['temp_file'], 'r') as temp_f:
                temp_f.copy('tensors', train_group)
            
            train_group.create_dataset('label_event', data=train_data['label_event'])
            train_group.create_dataset('label_mag', data=train_data['label_mag'])
            train_group.create_dataset('label_azm', data=train_data['label_azm'])
            train_group.create_dataset('cosmic_features', data=train_data['cosmic_features'])
            train_group.create_dataset('dates', data=train_data['dates'])
            train_group.attrs['num_stations'] = len(train_data['stations'])
            train_group.attrs['stations'] = train_data['stations']
            
            # Get shape from temp file
            with h5py.File(train_data['temp_file'], 'r') as temp_f:
                shape = temp_f['tensors'].shape
            print(f"   ✓ Train set saved: {shape}")
            
            # Delete temp file
            os.remove(train_data['temp_file'])
        else:
            print(f"   ⚠️ No train data to save")
        
        # Save val set
        if val_data is not None:
            val_group = f.create_group('val')
            
            # Copy tensors from temp file
            with h5py.File(val_data['temp_file'], 'r') as temp_f:
                temp_f.copy('tensors', val_group)
            
            val_group.create_dataset('label_event', data=val_data['label_event'])
            val_group.create_dataset('label_mag', data=val_data['label_mag'])
            val_group.create_dataset('label_azm', data=val_data['label_azm'])
            val_group.create_dataset('cosmic_features', data=val_data['cosmic_features'])
            val_group.create_dataset('dates', data=val_data['dates'])
            val_group.attrs['num_stations'] = len(val_data['stations'])
            val_group.attrs['stations'] = val_data['stations']
            
            # Get shape from temp file
            with h5py.File(val_data['temp_file'], 'r') as temp_f:
                shape = temp_f['tensors'].shape
            print(f"   ✓ Val set saved: {shape}")
            
            # Delete temp file
            os.remove(val_data['temp_file'])
        else:
            print(f"   ⚠️ No val data to save")
        
        # Global attributes
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['version'] = 'v10'
        f.attrs['format'] = 'multi-station-graph'
        f.attrs['train_end_date'] = TRAIN_END_DATE
        f.attrs['val_start_date'] = VAL_START_DATE
        f.attrs['val_end_date'] = VAL_END_DATE
        f.attrs['target_ratio'] = TARGET_RATIO
        f.attrs['min_magnitude'] = MIN_MAGNITUDE
    
    file_size = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"\n✓ File saved successfully!")
    print(f"   Size: {file_size:.2f} MB")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("DATA SCAVENGING & BALANCING FOR ANTIGRAVITY PROJECT")
    print("="*70)
    print(f"Target: Create balanced train/val sets with 1:{TARGET_RATIO} ratio")
    print(f"Exclude: All data >= {BLINDTEST_START_DATE}")
    print("="*70)
    
    # TAHAP 1: Scan and collect data
    df_master = scan_hdf5_files()
    
    if len(df_master) == 0:
        print("\n❌ No data found! Exiting.")
        return
    
    # TAHAP 2: Fix space weather
    df_master = fix_space_weather(df_master)
    
    # TAHAP 3: Temporal split
    df_train, df_val = temporal_split(df_master)
    
    # TAHAP 4: Balance datasets
    df_train_balanced = balance_dataset(df_train, 'train')
    df_val_balanced = balance_dataset(df_val, 'val')
    
    # TAHAP 5: Create multi-station graphs
    train_data = create_multistation_graphs(df_train_balanced, 'train') if len(df_train_balanced) > 0 else None
    val_data = create_multistation_graphs(df_val_balanced, 'val') if len(df_val_balanced) > 0 else None
    
    # TAHAP 6: Save final dataset
    save_dataset(train_data, val_data)
    
    # Final report
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    
    if train_data is not None:
        print(f"\n📊 TRAIN SET:")
        print(f"   Total days: {len(train_data['dates'])}")
        print(f"   Events: {train_data['label_event'].sum()} ({train_data['label_event'].sum()/len(train_data['label_event'])*100:.1f}%)")
        print(f"   Noise: {len(train_data['label_event']) - train_data['label_event'].sum()}")
        print(f"   Ratio: 1:{(len(train_data['label_event']) - train_data['label_event'].sum())/train_data['label_event'].sum():.1f}")
        print(f"   Magnitude range: {train_data['label_mag'][train_data['label_mag'] > 0].min():.2f} - {train_data['label_mag'].max():.2f}")
    else:
        print(f"\n⚠️ TRAIN SET: No data")
    
    if val_data is not None:
        print(f"\n📊 VALIDATION SET:")
        print(f"   Total days: {len(val_data['dates'])}")
        print(f"   Events: {val_data['label_event'].sum()} ({val_data['label_event'].sum()/len(val_data['label_event'])*100:.1f}%)")
        print(f"   Noise: {len(val_data['label_event']) - val_data['label_event'].sum()}")
        print(f"   Ratio: 1:{(len(val_data['label_event']) - val_data['label_event'].sum())/val_data['label_event'].sum():.1f}")
        print(f"   Magnitude range: {val_data['label_mag'][val_data['label_mag'] > 0].min():.2f} - {val_data['label_mag'].max():.2f}")
    else:
        print(f"\n⚠️ VALIDATION SET: No data")
    
    print(f"\n✅ Dataset creation complete!")
    print(f"   Output: {OUTPUT_FILE}")
    print("="*70)


if __name__ == '__main__':
    main()
