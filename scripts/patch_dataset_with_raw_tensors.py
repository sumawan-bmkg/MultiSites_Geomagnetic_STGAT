"""
Patch Dataset V10 with Raw Tensors from tensor/raw_tensors_v2/
=============================================================
Tujuan:
  - Scan folder tensor/raw_tensors_v2/ untuk noise samples
  - Tambahkan noise ke VAL set (2024-12-31 to 2025-03-31)
  - Tambahkan noise ke TRAIN set jika ada data <= 2023-12-31
  - Hasilkan dataset_v11_patched.h5 yang lebih seimbang
  - Target rasio: 1 Event : 3-4 Noise

Temuan dari scan:
  - raw_tensors_v2: 8,573 files, 21 stasiun
  - Date range: 2024-12-31 to 2026-03-30
  - Val noise tersedia: 1,791 files (91 hari, 2024-12-31 to 2025-03-31)
  - Blindtest (EXCLUDE): 1,599 files (>= 2026-01-01)
  - Semua file adalah background noise (label_event = 0)
"""

import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, date, timezone
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_TENSOR_DIR_V2 = 'tensor/raw_tensors_v2'
INPUT_DATASET    = 'dataset_v10_train_val_graphs.h5'
OUTPUT_DATASET   = 'dataset_v11_patched.h5'

# Date boundaries
TRAIN_END   = date(2023, 12, 31)
VAL_START   = date(2024,  1,  1)
VAL_END     = date(2025,  3, 31)
BLIND_START = date(2026,  1,  1)

# Balancing target
TARGET_NOISE_RATIO = 4   # 1 event : 4 noise (max)
MIN_MAGNITUDE      = 4.0

# Earthquake catalog for label lookup
EQ_CATALOG = 'intial/earthquake_catalog_2018_2025_merged_robust.csv'
STATION_FILE = 'intial/lokasi_stasiun.csv'


# ============================================================================
# STEP 1: INVENTORY RAW TENSORS
# ============================================================================

def inventory_raw_tensors():
    """Scan raw_tensors_v2 and build inventory DataFrame."""
    print("=" * 70)
    print("STEP 1: INVENTORY RAW TENSORS")
    print("=" * 70)

    records = []
    for fname in os.listdir(RAW_TENSOR_DIR_V2):
        if not fname.endswith('.npy') or fname.startswith('desktop'):
            continue
        parts = fname.replace('raw_tensor_', '').replace('.npy', '').split('_')
        if len(parts) != 2:
            continue
        station, ts_str = parts
        try:
            ts  = int(ts_str)
            dt  = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        except ValueError:
            continue
        records.append({
            'file':      fname,
            'path':      os.path.join(RAW_TENSOR_DIR_V2, fname),
            'station':   station,
            'timestamp': ts,
            'date':      dt,
        })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])

    print(f"  Total raw tensor files : {len(df)}")
    print(f"  Date range             : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Unique stations        : {sorted(df['station'].unique())}")

    # Split by date
    blind_mask = df['date'].dt.date >= BLIND_START
    val_mask   = (df['date'].dt.date >= VAL_START) & (df['date'].dt.date <= VAL_END)
    train_mask = df['date'].dt.date <= TRAIN_END

    df_val_noise   = df[val_mask].copy()
    df_train_noise = df[train_mask].copy()
    df_blind       = df[blind_mask].copy()

    print(f"\n  Val noise  (2024-01-01 to 2025-03-31) : {len(df_val_noise)} files, "
          f"{df_val_noise['date'].dt.date.nunique()} unique days")
    print(f"  Train noise (<= 2023-12-31)            : {len(df_train_noise)} files")
    print(f"  Blindtest  (>= 2026-01-01) EXCLUDED   : {len(df_blind)} files")

    return df_val_noise, df_train_noise


# ============================================================================
# STEP 2: LOAD EXISTING V10 DATASET
# ============================================================================

def load_v10_summary():
    """Print summary of existing V10 dataset."""
    print("\n" + "=" * 70)
    print("STEP 2: EXISTING V10 DATASET SUMMARY")
    print("=" * 70)

    with h5py.File(INPUT_DATASET, 'r') as f:
        for grp in ['train', 'val']:
            if grp not in f:
                print(f"  {grp}: NOT FOUND")
                continue
            g = f[grp]
            events = g['label_event'][:]
            n_ev   = int(events.sum())
            n_no   = len(events) - n_ev
            dates_raw = g['dates'][:]
            if isinstance(dates_raw[0], bytes):
                dates = [d.decode() for d in dates_raw]
            else:
                dates = list(dates_raw)
            print(f"\n  {grp.upper()} SET:")
            print(f"    Shape    : {g['tensors'].shape}")
            print(f"    Days     : {len(dates)}")
            print(f"    Events   : {n_ev}  ({n_ev/len(events)*100:.1f}%)")
            print(f"    Noise    : {n_no}  ({n_no/len(events)*100:.1f}%)")
            print(f"    Stations : {list(g.attrs.get('stations', []))}")
            print(f"    Date range: {min(dates)} to {max(dates)}")


# ============================================================================
# STEP 3: BUILD NOISE GRAPH SNAPSHOTS FROM RAW TENSORS
# ============================================================================

def build_noise_snapshots(df_noise, split_name='val'):
    """
    Group raw tensor files by date, build multi-station graph snapshots.
    Returns dict with tensors, labels, dates, stations.
    Writes to a temp HDF5 to avoid RAM overflow.
    """
    print(f"\n{'='*70}")
    print(f"STEP 3: BUILD NOISE SNAPSHOTS ({split_name.upper()})")
    print("=" * 70)

    if len(df_noise) == 0:
        print("  No noise data available.")
        return None

    # All stations present in this noise batch
    stations = sorted(df_noise['station'].unique())
    n_stations = len(stations)
    sta_idx = {s: i for i, s in enumerate(stations)}

    # Group by date
    df_noise = df_noise.copy()
    df_noise['date_only'] = df_noise['date'].dt.date
    grouped = df_noise.groupby('date_only')
    dates_sorted = sorted(grouped.groups.keys())
    n_days = len(dates_sorted)

    print(f"  Stations  : {n_stations} -> {stations}")
    print(f"  Unique days: {n_days}")
    print(f"  Date range : {min(dates_sorted)} to {max(dates_sorted)}")

    temp_file = f'temp_noise_{split_name}.h5'

    label_event = np.zeros(n_days, dtype=np.int8)    # all noise = 0
    label_mag   = np.zeros(n_days, dtype=np.float32)
    label_azm   = np.zeros((n_days, n_stations), dtype=np.float32)
    cosmic      = np.zeros((n_days, 2), dtype=np.float32)
    dates_out   = []

    with h5py.File(temp_file, 'w') as out_f:
        tensors_ds = out_f.create_dataset(
            'tensors',
            shape=(n_days, n_stations, 3, 128, 1440),
            dtype=np.float16,
            compression='gzip',
            chunks=(1, n_stations, 3, 128, 1440)
        )

        for day_idx, day in enumerate(dates_sorted):
            if day_idx % 50 == 0:
                print(f"    Processing day {day_idx+1}/{n_days} ({day})...")

            day_df = grouped.get_group(day)
            day_tensor = np.zeros((n_stations, 3, 128, 1440), dtype=np.float16)

            for _, row in day_df.iterrows():
                s_idx = sta_idx.get(row['station'])
                if s_idx is None:
                    continue
                try:
                    arr = np.load(row['path']).astype(np.float16)
                    if arr.shape == (3, 128, 1440):
                        day_tensor[s_idx] = arr
                except Exception as e:
                    pass  # leave as zeros

            tensors_ds[day_idx] = day_tensor
            dates_out.append(str(day))

    print(f"  ✓ Noise snapshots written to {temp_file}")

    return {
        'temp_file':    temp_file,
        'label_event':  label_event,
        'label_mag':    label_mag,
        'label_azm':    label_azm,
        'cosmic':       cosmic,
        'dates':        np.array(dates_out, dtype='S10'),
        'stations':     stations,
        'n_days':       n_days,
    }


# ============================================================================
# STEP 4: MERGE AND BALANCE
# ============================================================================

def merge_and_save(val_noise_data):
    """
    Merge existing V10 data with new noise snapshots.
    Apply balancing: keep all events, undersample noise to TARGET_NOISE_RATIO.
    Save as dataset_v11_patched.h5.
    """
    print(f"\n{'='*70}")
    print("STEP 4: MERGE, BALANCE & SAVE")
    print("=" * 70)

    with h5py.File(OUTPUT_DATASET, 'w') as out_f:

        # ---- TRAIN SET: copy from V10 as-is (no new train noise available) ----
        with h5py.File(INPUT_DATASET, 'r') as in_f:
            if 'train' in in_f:
                print("\n  Copying TRAIN set from V10 (unchanged)...")
                in_f.copy('train', out_f)
                train_events = in_f['train']['label_event'][:]
                print(f"    Days   : {len(train_events)}")
                print(f"    Events : {int(train_events.sum())} ({int(train_events.sum())/len(train_events)*100:.1f}%)")
                print(f"    Noise  : {len(train_events)-int(train_events.sum())}")

        # ---- VAL SET: merge V10 val + new noise ----
        print("\n  Building VAL set (V10 val + new noise)...")

        with h5py.File(INPUT_DATASET, 'r') as in_f:
            v10_val = in_f['val']
            v10_tensors   = v10_val['tensors']       # (377, 22, 3, 128, 1440)
            v10_events    = v10_val['label_event'][:]
            v10_mags      = v10_val['label_mag'][:]
            v10_azms      = v10_val['label_azm'][:]
            v10_cosmic    = v10_val['cosmic_features'][:]
            v10_dates_raw = v10_val['dates'][:]
            v10_stations  = list(v10_val.attrs.get('stations', []))
            if isinstance(v10_stations[0], bytes):
                v10_stations = [s.decode() for s in v10_stations]
            v10_dates = [d.decode() if isinstance(d, bytes) else d for d in v10_dates_raw]

            n_v10 = len(v10_events)
            n_v10_events = int(v10_events.sum())
            n_v10_noise  = n_v10 - n_v10_events

            print(f"    V10 val: {n_v10} days, {n_v10_events} events, {n_v10_noise} noise")

            # New noise from raw tensors
            n_new_noise = val_noise_data['n_days'] if val_noise_data else 0
            print(f"    New noise from raw tensors: {n_new_noise} days")

            # Determine how many noise days to keep (balance)
            target_noise = min(n_v10_events * TARGET_NOISE_RATIO,
                               n_v10_noise + n_new_noise)
            print(f"    Target noise (1:{TARGET_NOISE_RATIO}): {target_noise} days")

            # Decide which V10 noise days to keep
            v10_event_idx = [i for i, e in enumerate(v10_events) if e == 1]
            v10_noise_idx = [i for i, e in enumerate(v10_events) if e == 0]

            # How many noise slots remain after V10 events
            noise_budget = target_noise
            # First fill with V10 noise (already in val period)
            keep_v10_noise = v10_noise_idx[:noise_budget]
            noise_budget -= len(keep_v10_noise)

            # Remaining budget filled from new raw tensor noise
            new_noise_days_to_use = min(noise_budget, n_new_noise)
            print(f"    V10 noise kept  : {len(keep_v10_noise)}")
            print(f"    New noise added : {new_noise_days_to_use}")

            # Indices to include from V10
            keep_v10_idx = sorted(v10_event_idx + keep_v10_noise)

            # Unified station list
            new_stations = val_noise_data['stations'] if val_noise_data else []
            all_stations = sorted(set(v10_stations) | set(new_stations))
            n_all = len(all_stations)
            sta_map_v10 = {s: all_stations.index(s) for s in v10_stations if s in all_stations}
            sta_map_new = {s: all_stations.index(s) for s in new_stations if s in all_stations}

            print(f"    Unified stations: {n_all} -> {all_stations}")

            total_days = len(keep_v10_idx) + new_noise_days_to_use
            print(f"    Total val days  : {total_days}")

            # Create output val group
            val_grp = out_f.create_group('val')
            tensors_ds = val_grp.create_dataset(
                'tensors',
                shape=(total_days, n_all, 3, 128, 1440),
                dtype=np.float16,
                compression='gzip',
                chunks=(1, n_all, 3, 128, 1440)
            )

            out_events = []
            out_mags   = []
            out_azms   = []
            out_cosmic = []
            out_dates  = []

            # --- Copy V10 val days (events + selected noise) ---
            print("    Writing V10 val days...")
            for write_idx, src_idx in enumerate(keep_v10_idx):
                day_tensor = np.zeros((n_all, 3, 128, 1440), dtype=np.float16)
                src_tensor = v10_tensors[src_idx]  # (22, 3, 128, 1440)
                for s, s_i in sta_map_v10.items():
                    old_i = v10_stations.index(s)
                    if old_i < src_tensor.shape[0]:
                        day_tensor[s_i] = src_tensor[old_i]
                tensors_ds[write_idx] = day_tensor

                # Azimuth: remap
                azm_row = np.zeros(n_all, dtype=np.float32)
                src_azm = v10_azms[src_idx]
                for s, s_i in sta_map_v10.items():
                    old_i = v10_stations.index(s)
                    if old_i < len(src_azm):
                        azm_row[s_i] = src_azm[old_i]

                out_events.append(v10_events[src_idx])
                out_mags.append(v10_mags[src_idx])
                out_azms.append(azm_row)
                out_cosmic.append(v10_cosmic[src_idx])
                out_dates.append(v10_dates[src_idx])

            # --- Append new noise days ---
            if val_noise_data and new_noise_days_to_use > 0:
                print(f"    Writing {new_noise_days_to_use} new noise days...")
                with h5py.File(val_noise_data['temp_file'], 'r') as noise_f:
                    noise_tensors = noise_f['tensors']
                    for ni in range(new_noise_days_to_use):
                        write_idx = len(keep_v10_idx) + ni
                        day_tensor = np.zeros((n_all, 3, 128, 1440), dtype=np.float16)
                        src_tensor = noise_tensors[ni]  # (n_new_stations, 3, 128, 1440)
                        for s, s_i in sta_map_new.items():
                            old_i = new_stations.index(s)
                            if old_i < src_tensor.shape[0]:
                                day_tensor[s_i] = src_tensor[old_i]
                        tensors_ds[write_idx] = day_tensor

                        azm_row = np.zeros(n_all, dtype=np.float32)
                        out_events.append(val_noise_data['label_event'][ni])
                        out_mags.append(val_noise_data['label_mag'][ni])
                        out_azms.append(azm_row)
                        out_cosmic.append(val_noise_data['cosmic'][ni])
                        out_dates.append(val_noise_data['dates'][ni].decode()
                                         if isinstance(val_noise_data['dates'][ni], bytes)
                                         else val_noise_data['dates'][ni])

            # Save labels
            val_grp.create_dataset('label_event',    data=np.array(out_events, dtype=np.int8))
            val_grp.create_dataset('label_mag',      data=np.array(out_mags,   dtype=np.float32))
            val_grp.create_dataset('label_azm',      data=np.array(out_azms,   dtype=np.float32))
            val_grp.create_dataset('cosmic_features',data=np.array(out_cosmic, dtype=np.float32))
            val_grp.create_dataset('dates',          data=np.array(out_dates,  dtype='S10'))
            val_grp.attrs['num_stations'] = n_all
            val_grp.attrs['stations']     = all_stations

        # Global attrs
        out_f.attrs['created']         = datetime.now().isoformat()
        out_f.attrs['version']         = 'v11'
        out_f.attrs['format']          = 'multi-station-graph'
        out_f.attrs['train_end_date']  = str(TRAIN_END)
        out_f.attrs['val_start_date']  = str(VAL_START)
        out_f.attrs['val_end_date']    = str(VAL_END)
        out_f.attrs['target_ratio']    = TARGET_NOISE_RATIO
        out_f.attrs['min_magnitude']   = MIN_MAGNITUDE
        out_f.attrs['source_v10']      = INPUT_DATASET
        out_f.attrs['raw_tensor_dir']  = RAW_TENSOR_DIR_V2

    size_mb = os.path.getsize(OUTPUT_DATASET) / (1024**2)
    print(f"\n  ✓ Saved: {OUTPUT_DATASET}  ({size_mb:.2f} MB)")


# ============================================================================
# STEP 5: FINAL REPORT
# ============================================================================

def final_report():
    print(f"\n{'='*70}")
    print("STEP 5: FINAL REPORT")
    print("=" * 70)

    with h5py.File(OUTPUT_DATASET, 'r') as f:
        for grp in ['train', 'val']:
            if grp not in f:
                continue
            g = f[grp]
            events = g['label_event'][:]
            n_ev = int(events.sum())
            n_no = len(events) - n_ev
            ratio = n_no / n_ev if n_ev > 0 else float('inf')
            dates_raw = g['dates'][:]
            dates = [d.decode() if isinstance(d, bytes) else d for d in dates_raw]
            stations = list(g.attrs.get('stations', []))
            if stations and isinstance(stations[0], bytes):
                stations = [s.decode() for s in stations]

            print(f"\n  {grp.upper()} SET:")
            print(f"    Shape     : {g['tensors'].shape}")
            print(f"    Days      : {len(dates)}")
            print(f"    Events    : {n_ev}  ({n_ev/len(events)*100:.1f}%)")
            print(f"    Noise     : {n_no}  ({n_no/len(events)*100:.1f}%)")
            print(f"    Ratio     : 1:{ratio:.1f}")
            print(f"    Stations  : {len(stations)} -> {stations}")
            print(f"    Date range: {min(dates)} to {max(dates)}")

            # NaN check
            sample = g['tensors'][:5]
            print(f"    NaN check (first 5 days): {np.isnan(sample).sum()} NaN values")

    print(f"\n  Output file : {OUTPUT_DATASET}")
    print(f"  File size   : {os.path.getsize(OUTPUT_DATASET)/(1024**2):.2f} MB")
    print("\n" + "=" * 70)
    print("✅ PATCHING COMPLETE")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("PATCH DATASET WITH RAW TENSORS (V10 -> V11)")
    print("=" * 70)
    print(f"  Input  : {INPUT_DATASET}")
    print(f"  Output : {OUTPUT_DATASET}")
    print(f"  Raw tensors dir: {RAW_TENSOR_DIR_V2}")
    print(f"  Target ratio: 1:{TARGET_NOISE_RATIO}")

    # Step 1: Inventory
    df_val_noise, df_train_noise = inventory_raw_tensors()

    # Step 2: V10 summary
    load_v10_summary()

    # Step 3: Build noise snapshots for val
    val_noise_data = build_noise_snapshots(df_val_noise, split_name='val')

    # Step 4: Merge and save
    merge_and_save(val_noise_data)

    # Cleanup temp files
    if val_noise_data and os.path.exists(val_noise_data['temp_file']):
        os.remove(val_noise_data['temp_file'])
        print(f"\n  Cleaned up temp file: {val_noise_data['temp_file']}")

    # Step 5: Final report
    final_report()


if __name__ == '__main__':
    main()
