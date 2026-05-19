import h5py
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Configuration
HDF5_FILE = 'dataset_v13_train_val_M5_patched.h5'
CATALOGS = [
    'intial/earthquake_catalog_2018_2025_merged.csv',
    'intial/earthquake_catalog_2018_2025_merged_robust.csv'
]

def patch_magnitudes():
    print("="*70)
    print("ANTIGRAVITY - IN-PLACE MAGNITUDE RESTORATION")
    print("="*70)

    # 1. Muat Katalog-katalog dan buat Lookup Table
    all_dfs = []
    for cat in CATALOGS:
        if not os.path.exists(cat): continue
        print(f"Loading catalog from {cat}...")
        df = pd.read_csv(cat)
        
        # Standarisasi kolom datetime
        if 'datetime' in df.columns:
            df['dt_raw'] = df['datetime']
        elif 'Date time' in df.columns:
            df['dt_raw'] = df['Date time']
        else:
            continue
            
        # Standarisasi kolom Magnitude
        if 'Magnitude' in df.columns:
            df['mag_raw'] = df['Magnitude']
        else:
            continue
            
        all_dfs.append(df[['dt_raw', 'mag_raw']])
    
    full_df = pd.concat(all_dfs)
    
    # Ekstrak tanggal (YYYY-MM-DD) dengan cerdas
    def extract_date(dt_str):
        if pd.isna(dt_str): return None
        # Bersihkan karakter non-digit/dash di awal jika ada
        s = str(dt_str).strip()
        # Ambil 10 karakter pertama (YYYY-MM-DD)
        return s[:10]

    full_df['date_key'] = full_df['dt_raw'].apply(extract_date)
    
    # Buat lookup table: date -> max Magnitude
    lookup = full_df.groupby('date_key')['mag_raw'].max().to_dict()
    print(f"Lookup table created with {len(lookup)} unique dates.")

    # 2. Buka HDF5 dalam mode read/write ('r+')
    if not os.path.exists(HDF5_FILE):
        print(f"ERROR: File {HDF5_FILE} not found.")
        return

    print(f"Opening HDF5 file {HDF5_FILE} in 'r+' mode...")
    with h5py.File(HDF5_FILE, 'r+') as f:
        for split in ['train', 'val']:
            if split not in f: continue
            
            print(f"\nProcessing split: {split.upper()}")
            group = f[split]
            dates = group['dates'][:]
            events = group['label_event'][:]
            mags = group['label_mag'] # Dataset object for in-place write
            
            patch_count = 0
            missing_count = 0
            
            for i in range(len(events)):
                if events[i] == 1:
                    d_str = dates[i].decode() if isinstance(dates[i], bytes) else dates[i]
                    # Dates in HDF5 are YYYY-MM-DD
                    date_key = d_str.split(' ')[0]
                    
                    if date_key in lookup:
                        real_mag = lookup[date_key]
                        mags[i] = real_mag
                        patch_count += 1
                    else:
                        missing_count += 1
                        # print(f"  WARNING: No catalog match for {date_key}")
            
            print(f"  Split {split.upper()} finished.")
            print(f"  Patched: {patch_count} | Missing in Catalog: {missing_count}")

    # 3. Verifikasi Akhir
    print("\n" + "="*70)
    print("VERIFIKASI AKHIR")
    print("="*70)
    with h5py.File(HDF5_FILE, 'r') as f:
        for split in ['train', 'val']:
            if split not in f: continue
            m = f[split]['label_mag'][:]
            e = f[split]['label_event'][:]
            event_mags = m[e == 1]
            unique_mags = np.unique(event_mags)
            print(f"\nSplit {split.upper()}:")
            print(f"  Unique magnitudes count: {len(unique_mags)}")
            print(f"  Top 10 unique magnitudes: {list(np.round(unique_mags[:10], 2))}")
            
            if len(unique_mags) > 1:
                print(f"  STATUS: SUCCESS (Variasi ditemukan)")
            else:
                print(f"  STATUS: FAILED (Masih seragam {unique_mags})")

    print("\nDone!")

if __name__ == '__main__':
    patch_magnitudes()
