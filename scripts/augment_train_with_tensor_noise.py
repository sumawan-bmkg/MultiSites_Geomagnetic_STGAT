"""
Augment Train Set with Noise from tensor/processed_tensors_v2_60pct_snapshot/train/normal/
=========================================================================================
This script adds noise samples from the processed tensor folder to the V13 train set
to address the severe class imbalance (99.4% events, 0.6% noise).

WARNING: The noise data comes from 2025, which violates strict chronological split.
This is a pragmatic trade-off for training a usable model.
Use synthetic augmentation if chronological purity is required.

Output: dataset_v14_train_val_M5_augmented.h5
"""
import os
import re
import h5py
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

SRC_DATASET = 'dataset_v13_train_val_M5.h5'
OUT_DATASET = 'dataset_v14_train_val_M5_augmented.h5'
NOISE_FOLDER = r'D:\multi\1multisite\tensor\processed_tensors_v2_60pct_snapshot\train\normal'
MAX_NOISE_DAYS = 300  # cap to avoid exploding dataset size

def scan_noise_files(folder):
    """Scan noise .npy files and group by (date, station)."""
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    daily = defaultdict(dict)  # date_str -> {station_code: filepath}
    for fname in files:
        # e.g. event_ALR_20250101.npy
        m = re.match(r'event_(\w+)_(\d{8})\.npy', fname)
        if not m:
            continue
        station = m.group(1)
        date_str = m.group(2)
        daily[date_str][station] = os.path.join(folder, fname)
    return daily

def build_station_index_from_dataset(h5_file):
    """Extract station order from an existing dataset's attributes or val set."""
    with h5py.File(h5_file, 'r', locking=False) as h5:
        # Try to infer from val set dimensions if station names not stored
        # We'll use the 24-station order from the patched file if possible,
        # but for v13_train_val_M5 we can just use alphabetical + known list
        # Full 24-station list matching V13 dataset (alphabetical order used in V13)
        all_stations = [
            'ALR','AMB','CLP','GSI','GTO','JYP','KPY','LPS','LUT','LWA','LWK',
            'MLB','PLU','SBG','SCN','SKB','SMI','SRG','SRO','TND','TNT','TRD','TRT','YOG'
        ]
        # v13 uses 24 stations padded; we replicate the same alphabetical order + padding
        return all_stations

def load_existing_labels(src_file):
    """Load only small metadata arrays (labels, dates, cosmic) into RAM."""
    data = {}
    with h5py.File(src_file, 'r', locking=False) as h5:
        for split in ['train', 'val']:
            if split not in h5:
                continue
            grp = h5[split]
            data[split] = {
                'label_event': grp['label_event'][:],
                'label_mag': grp['label_mag'][:],
                'label_azm': grp['label_azm'][:],
                'label_dist': grp['label_dist'][:],
                'cosmic_features': grp['cosmic_features'][:],
                'dates': grp['dates'][:],
                'tensors_shape': grp['tensors'].shape,
                'tensors_dtype': grp['tensors'].dtype,
            }
    return data

def copy_dataset_chunked(src_grp, dst_grp, name, chunk_days=20):
    """Copy an HDF5 dataset from src to dst in day-chunks to avoid RAM overload."""
    src_ds = src_grp[name]
    shape = src_ds.shape
    dtype = src_ds.dtype
    # Determine chunk size along first axis
    if len(shape) == 5:
        chunk_shape = (chunk_days, shape[1], shape[2], shape[3], shape[4])
    elif len(shape) == 2:
        chunk_shape = (chunk_days, shape[1])
    elif len(shape) == 1:
        chunk_shape = (chunk_days,)
    else:
        chunk_shape = shape
    dst_ds = dst_grp.create_dataset(name, shape=shape, dtype=dtype, compression='gzip', chunks=chunk_shape)
    n = shape[0]
    for i in range(0, n, chunk_days):
        end = min(i + chunk_days, n)
        dst_ds[i:end] = src_ds[i:end]
        if (i // chunk_days) % 10 == 0:
            print(f'    ... copied {name} {end}/{n}')
    return dst_ds

def build_noise_snapshot(station_files, all_stations):
    """Build a single-day (24, 3, 128, 1440) tensor from station .npy files."""
    n_sta = len(all_stations)
    snapshot = np.zeros((n_sta, 3, 128, 1440), dtype=np.float16)
    sta_idx = {s: i for i, s in enumerate(all_stations)}
    for sta_code, fpath in station_files.items():
        if sta_code not in sta_idx:
            continue
        try:
            arr = np.load(fpath)
            if arr.shape != (3, 128, 1440):
                # Resize if needed
                arr = np.resize(arr, (3, 128, 1440))
            snapshot[sta_idx[sta_code]] = arr.astype(np.float16)
        except Exception as e:
            print(f'    Warning: failed to load {fpath}: {e}')
    return snapshot

def main():
    print('=' * 70)
    print('AUGMENT TRAIN SET WITH NOISE FROM tensor/ FOLDER')
    print('=' * 70)

    if not os.path.exists(SRC_DATASET):
        print(f'ERROR: Source dataset not found: {SRC_DATASET}')
        return
    if not os.path.exists(NOISE_FOLDER):
        print(f'ERROR: Noise folder not found: {NOISE_FOLDER}')
        return

    # 1. Load only lightweight metadata
    print('\n[1] Loading existing dataset metadata...')
    existing = load_existing_labels(SRC_DATASET)
    train_meta = existing['train']
    val_meta = existing['val']
    n_train_old = len(train_meta['dates'])
    n_val = len(val_meta['dates'])
    print(f'  Train: {n_train_old} days, tensor shape {train_meta["tensors_shape"]}, dtype {train_meta["tensors_dtype"]}')
    print(f'  Val:   {n_val} days, tensor shape {val_meta["tensors_shape"]}, dtype {val_meta["tensors_dtype"]}')

    # 2. Scan noise files
    print('\n[2] Scanning noise files...')
    daily_noise = scan_noise_files(NOISE_FOLDER)
    print(f'  Unique noise days found: {len(daily_noise)}')
    sorted_dates = sorted(daily_noise.keys())[:MAX_NOISE_DAYS]
    n_noise = len(sorted_dates)
    print(f'  Using first {n_noise} days')

    all_stations = build_station_index_from_dataset(SRC_DATASET)
    n_sta = len(all_stations)
    print(f'  Station list ({n_sta}): {all_stations}')

    # 3. Prepare output file with pre-allocated datasets
    print('\n[3] Preparing output file...')
    n_train_new = n_train_old + n_noise
    chunk_days = 20
    tensor_chunks = (chunk_days, n_sta, 3, 128, 1440)

    with h5py.File(SRC_DATASET, 'r', locking=False) as src, \
         h5py.File(OUT_DATASET, 'w') as out:

        # ---------- TRAIN group (original + noise) ----------
        tg = out.create_group('train')
        # Tensors: pre-allocate
        train_t = tg.create_dataset('tensors', shape=(n_train_new, n_sta, 3, 128, 1440),
                                      dtype=train_meta['tensors_dtype'],
                                      compression='gzip', chunks=tensor_chunks)
        # Small arrays: concatenate in memory (they are tiny)
        noise_dates_arr = np.array([d.encode('ascii') for d in sorted_dates])
        merged_dates = np.concatenate([train_meta['dates'], noise_dates_arr])
        merged_event = np.concatenate([train_meta['label_event'], np.zeros(n_noise, dtype=np.int8)])
        merged_mag = np.concatenate([train_meta['label_mag'], np.zeros(n_noise, dtype=np.float32)])
        merged_azm = np.concatenate([train_meta['label_azm'], np.zeros((n_noise, n_sta), dtype=np.float32)], axis=0)
        merged_dist = np.concatenate([train_meta['label_dist'], np.zeros((n_noise, n_sta), dtype=np.float32)], axis=0)
        avg_cosmic = train_meta['cosmic_features'].mean(axis=0) if n_train_old > 0 else np.array([5.0, -175.0], dtype=np.float32)
        noise_cosmic = np.tile(avg_cosmic, (n_noise, 1)).astype(np.float32)
        merged_cosmic = np.concatenate([train_meta['cosmic_features'], noise_cosmic], axis=0)

        tg.create_dataset('dates', data=merged_dates)
        tg.create_dataset('label_event', data=merged_event)
        tg.create_dataset('label_mag', data=merged_mag)
        tg.create_dataset('label_azm', data=merged_azm)
        tg.create_dataset('label_dist', data=merged_dist)
        tg.create_dataset('cosmic_features', data=merged_cosmic)

        # Copy original train tensors in chunks
        print('  Copying original train tensors...')
        src_train_t = src['train/tensors']
        for i in range(0, n_train_old, chunk_days):
            end = min(i + chunk_days, n_train_old)
            train_t[i:end] = src_train_t[i:end]
            if (i // chunk_days) % 5 == 0:
                print(f'    ... copied train {end}/{n_train_old}')

        # Build and append noise tensors in batches
        print('  Building and appending noise tensors...')
        batch_size = 50
        for batch_start in range(0, n_noise, batch_size):
            batch_end = min(batch_start + batch_size, n_noise)
            batch_snapshots = []
            for idx in range(batch_start, batch_end):
                date_str = sorted_dates[idx]
                sta_files = daily_noise[date_str]
                snap = build_noise_snapshot(sta_files, all_stations)
                batch_snapshots.append(snap)
            batch_arr = np.array(batch_snapshots, dtype=train_meta['tensors_dtype'])
            write_start = n_train_old + batch_start
            write_end = n_train_old + batch_end
            train_t[write_start:write_end] = batch_arr
            print(f'    ... appended noise {write_end - n_train_old}/{n_noise}')

        # ---------- VAL group (copy as-is, chunked) ----------
        print('  Copying validation data...')
        vg = out.create_group('val')
        copy_dataset_chunked(src['val'], vg, 'tensors', chunk_days=20)
        copy_dataset_chunked(src['val'], vg, 'dates', chunk_days=50)
        copy_dataset_chunked(src['val'], vg, 'label_event', chunk_days=50)
        copy_dataset_chunked(src['val'], vg, 'label_mag', chunk_days=50)
        copy_dataset_chunked(src['val'], vg, 'label_azm', chunk_days=20)
        copy_dataset_chunked(src['val'], vg, 'label_dist', chunk_days=20)
        copy_dataset_chunked(src['val'], vg, 'cosmic_features', chunk_days=50)

        # ---------- Attributes ----------
        n_total = n_train_new
        n_ev = int(merged_event.sum())
        n_no = n_total - n_ev
        out.attrs['version'] = 'v14_augmented'
        out.attrs['source'] = SRC_DATASET
        out.attrs['noise_source'] = NOISE_FOLDER
        out.attrs['noise_days_added'] = n_noise
        out.attrs['train_event_ratio'] = float(n_ev / n_total)
        out.attrs['train_noise_ratio'] = float(n_no / n_total)
        out.attrs['train_end'] = '2023-12-31 (original) + 2025 noise (chronological warning)'
        out.attrs['val_start'] = '2024-01-01'
        out.attrs['val_end'] = '2025-03-31'
        out.attrs['mw_threshold'] = 5.0
        out.attrs['warning'] = 'Noise added from 2025 data violates strict chronological split. Use synthetic augmentation if purity is required.'

    size_mb = os.path.getsize(OUT_DATASET) / (1024**2)
    print('\n[4] Summary')
    print(f'  New train set: {n_total} days')
    print(f'    Events: {n_ev} ({100*n_ev/n_total:.1f}%)')
    print(f'    Noise:  {n_no} ({100*n_no/n_total:.1f}%)')
    print(f'    Ratio (noise:event): 1:{n_no/n_ev:.1f}')
    print(f'  Saved: {OUT_DATASET} ({size_mb:.1f} MB)')
    print('\n' + '=' * 70)
    print('DONE')
    print('=' * 70)

if __name__ == '__main__':
    main()
