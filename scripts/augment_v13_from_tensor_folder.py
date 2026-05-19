"""
Augment V13 Dataset with Data from tensor/ Folder
==================================================
Temuan investigasi:
  1. tensor/raw_tensors_v2/         - 8,573 noise files (2024-12-31 to 2026-03-30)
  2. tensor/processed_tensors_v2/   - 6,956 labeled files (2025 only, float32)
  3. tensor/processed_tensors_v2_60pct_snapshot/ - 8,010 labeled files (2025 only, float64)

Strategi penambahan:
  A. VAL SET: Tambahkan noise dari processed_tensors_v2/train/normal
     (90 hari, 2025-01-01 to 2025-03-31, 20 stasiun, float32)
     -> Meningkatkan kualitas tensor val noise (data nyata, bukan proxy)
  
  B. VAL SET: Tambahkan noise dari raw_tensors_v2
     (455 hari, 2024-12-31 to 2026-03-30, 21 stasiun)
     -> Sudah dilakukan di V11, konfirmasi coverage
  
  C. TRAIN SET: Tidak ada data baru untuk train (semua 2025, bukan <=2023)
     -> Train tetap menggunakan proxy dari source HDF5

Output: dataset_v13_augmented.h5
"""

import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, date, timezone
from math import radians, sin, cos, sqrt, atan2, degrees
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

INPUT_FILE   = 'dataset_v13_train_val_M5_patched.h5'
OUTPUT_FILE  = 'dataset_v13_augmented.h5'
STATION_FILE = 'intial/lokasi_stasiun.csv'

PROC_V2_NORMAL_TRAIN = 'tensor/processed_tensors_v2/train/normal'
RAW_TENSOR_DIR       = 'tensor/raw_tensors_v2'

MW_THRESHOLD = 5.0
VAL_START    = date(2024, 1, 1)
VAL_END      = date(2025, 3, 31)
BLIND_START  = date(2026, 1, 1)
TARGET_RATIO = 4


# ============================================================================
# HELPERS
# ============================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def azimuth_calc(lat1, lon1, lat2, lon2):
    phi1, phi2 = radians(lat1), radians(lat2)
    dlam = radians(lon2 - lon1)
    x = sin(dlam) * cos(phi2)
    y = cos(phi1)*sin(phi2) - sin(phi1)*cos(phi2)*cos(dlam)
    return (degrees(atan2(x, y)) + 360) % 360


def load_stations():
    stations = {}
    with open(STATION_FILE, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.replace('\xa0', '').replace('\u00a0', '')
        parts = line.strip().split(';')
        if len(parts) >= 3 and parts[0].strip() and parts[1].strip() and parts[2].strip():
            code = parts[0].strip()
            try:
                lat = float(parts[1].strip())
                lon = float(parts[2].strip())
                stations[code] = (lat, lon)
            except ValueError:
                pass
    return stations


def load_dst_lookup():
    dst_lines = []
    with open('intial/dst.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('DATE') or line.startswith('|'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    dst_lines.append({'date': parts[0], 'dst': float(parts[3])})
                except Exception:
                    pass
    dst_df = pd.DataFrame(dst_lines)
    dst_df['dt'] = pd.to_datetime(dst_df['date'])
    dst_df['date_str'] = dst_df['dt'].dt.strftime('%Y-%m-%d')
    return dst_df.groupby('date_str')['dst'].mean().to_dict()


# ============================================================================
# STEP 1: INVENTORY NEW NOISE DATA FROM tensor/
# ============================================================================

def inventory_new_noise():
    """Find all new noise tensors from tensor/ folder for val period."""
    print("=" * 70)
    print("STEP 1: INVENTORY NEW NOISE DATA FROM tensor/")
    print("=" * 70)

    # Source A: processed_tensors_v2/train/normal (val period only)
    noise_a = {}  # {date_str: {station: filepath}}
    if os.path.isdir(PROC_V2_NORMAL_TRAIN):
        files = [f for f in os.listdir(PROC_V2_NORMAL_TRAIN) if f.endswith('.npy')]
        for fname in files:
            parts = fname.replace('.npy', '').split('_')
            if len(parts) >= 3:
                station  = parts[1]
                date_str = parts[2]
                try:
                    dt = datetime.strptime(date_str, '%Y%m%d').date()
                    if VAL_START <= dt <= VAL_END:
                        if date_str not in noise_a:
                            noise_a[date_str] = {}
                        noise_a[date_str][station] = os.path.join(PROC_V2_NORMAL_TRAIN, fname)
                except Exception:
                    pass

    print(f"\n  Source A (processed_tensors_v2/train/normal, val period):")
    print(f"    Dates: {len(noise_a)}")
    if noise_a:
        dates_a = sorted(noise_a.keys())
        print(f"    Range: {dates_a[0]} to {dates_a[-1]}")
        stations_a = set(s for d in noise_a.values() for s in d.keys())
        print(f"    Stations: {len(stations_a)} -> {sorted(stations_a)}")

    # Source B: raw_tensors_v2 (val period only, exclude blindtest)
    noise_b = {}
    if os.path.isdir(RAW_TENSOR_DIR):
        for fname in os.listdir(RAW_TENSOR_DIR):
            if not fname.endswith('.npy') or fname.startswith('desktop'):
                continue
            parts = fname.replace('raw_tensor_', '').replace('.npy', '').split('_')
            if len(parts) == 2:
                station, ts_str = parts
                try:
                    ts = int(ts_str)
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
                    if VAL_START <= dt <= VAL_END:
                        date_str = dt.strftime('%Y%m%d')
                        if date_str not in noise_b:
                            noise_b[date_str] = {}
                        noise_b[date_str][station] = os.path.join(RAW_TENSOR_DIR, fname)
                except Exception:
                    pass

    print(f"\n  Source B (raw_tensors_v2, val period):")
    print(f"    Dates: {len(noise_b)}")
    if noise_b:
        dates_b = sorted(noise_b.keys())
        print(f"    Range: {dates_b[0]} to {dates_b[-1]}")
        stations_b = set(s for d in noise_b.values() for s in d.keys())
        print(f"    Stations: {len(stations_b)} -> {sorted(stations_b)}")

    # Merge: prefer Source A (processed, higher quality) over Source B (raw)
    merged = {}
    for date_str, sta_dict in noise_b.items():
        merged[date_str] = dict(sta_dict)
    for date_str, sta_dict in noise_a.items():
        if date_str not in merged:
            merged[date_str] = {}
        merged[date_str].update(sta_dict)  # A overrides B

    print(f"\n  Merged (A+B, A preferred):")
    print(f"    Total dates: {len(merged)}")

    return merged


# ============================================================================
# STEP 2: CHECK WHAT'S ALREADY IN V13 VAL
# ============================================================================

def check_existing_val():
    """Check current V13 val set coverage."""
    print("\n" + "=" * 70)
    print("STEP 2: CURRENT V13 VAL SET COVERAGE")
    print("=" * 70)

    with h5py.File(INPUT_FILE, 'r') as f:
        g = f['val']
        events    = g['label_event'][:]
        dates_raw = g['dates'][:]
        dates     = [d.decode() if isinstance(d, bytes) else d for d in dates_raw]
        tensors   = g['tensors']

        n_ev = int(events.sum())
        n_no = len(events) - n_ev

        # Check tensor quality
        noise_idx  = [i for i, e in enumerate(events) if e == 0]
        event_idx  = [i for i, e in enumerate(events) if e == 1]

        noise_nonzero = sum(1 for i in noise_idx if np.any(tensors[i] != 0))
        event_nonzero = sum(1 for i in event_idx if np.any(tensors[i] != 0))

        print(f"\n  Val set: {len(dates)} days")
        print(f"  Events: {n_ev} | Noise: {n_no}")
        print(f"  Noise days with real tensors: {noise_nonzero}/{n_no}")
        print(f"  Event days with real tensors: {event_nonzero}/{n_ev}")

        # Get existing noise dates
        existing_noise_dates = set(
            dates[i].replace('-', '') for i in noise_idx
        )

    return existing_noise_dates


# ============================================================================
# STEP 3: BUILD AUGMENTED VAL SET
# ============================================================================

def build_augmented_val(new_noise_lookup, existing_noise_dates,
                        station_coords, all_stations, dst_lookup):
    """
    Rebuild val set with better tensors from tensor/ folder.
    For noise days that have real tensors in tensor/, replace the proxy.
    Also add new noise days if available.
    """
    print("\n" + "=" * 70)
    print("STEP 3: BUILD AUGMENTED VAL SET")
    print("=" * 70)

    n_sta   = len(all_stations)
    sta_idx = {s: i for i, s in enumerate(all_stations)}

    with h5py.File(INPUT_FILE, 'r') as src:
        g_val = src['val']
        events_src    = g_val['label_event'][:]
        mags_src      = g_val['label_mag'][:]
        azms_src      = g_val['label_azm'][:]
        dists_src     = g_val['label_dist'][:]
        cosmic_src    = g_val['cosmic_features'][:]
        dates_src_raw = g_val['dates'][:]
        dates_src     = [d.decode() if isinstance(d, bytes) else d for d in dates_src_raw]
        tensors_src   = g_val['tensors']
        src_stations  = list(g_val.attrs.get('stations', []))
        if src_stations and isinstance(src_stations[0], bytes):
            src_stations = [s.decode() for s in src_stations]

        n_days = len(dates_src)
        print(f"\n  Source val: {n_days} days, {int(events_src.sum())} events, {n_days - int(events_src.sum())} noise")

        # Find new noise dates not yet in val
        existing_dates_set = set(d.replace('-', '') for d in dates_src)
        new_dates = sorted([
            d for d in new_noise_lookup.keys()
            if d not in existing_dates_set
        ])
        print(f"  New noise dates to add: {len(new_dates)}")

        # Determine how many to add (balance: target 1:4 ratio)
        n_ev = int(events_src.sum())
        n_no = n_days - n_ev
        target_noise = n_ev * TARGET_RATIO
        can_add = max(0, target_noise - n_no)
        n_add = min(len(new_dates), can_add)
        print(f"  Target noise: {target_noise} | Current: {n_no} | Can add: {n_add}")

        new_dates_to_add = new_dates[:n_add]
        total_out = n_days + n_add

        print(f"  Output val: {total_out} days ({n_ev} events, {n_no + n_add} noise)")

        # Write output
        temp_file = 'temp_augmented_val.h5'
        events_new  = np.zeros(total_out, dtype=np.int8)
        mags_new    = np.zeros(total_out, dtype=np.float32)
        azms_new    = np.zeros((total_out, n_sta), dtype=np.float32)
        dists_new   = np.zeros((total_out, n_sta), dtype=np.float32)
        cosmic_new  = np.zeros((total_out, 2), dtype=np.float32)
        dates_out   = []

        n_tensor_upgraded = 0
        n_tensor_new      = 0

        with h5py.File(temp_file, 'w') as tmp:
            tds = tmp.create_dataset(
                'tensors',
                shape=(total_out, n_sta, 3, 128, 1440),
                dtype=np.float16,
                compression='gzip',
                chunks=(1, n_sta, 3, 128, 1440)
            )

            # --- Copy existing val days (with tensor upgrade where possible) ---
            print(f"\n  Processing {n_days} existing val days...")
            for i in range(n_days):
                date_str = dates_src[i].replace('-', '')
                date_fmt = dates_src[i]

                # Copy labels
                events_new[i]  = events_src[i]
                mags_new[i]    = mags_src[i]
                azms_new[i]    = azms_src[i]
                dists_new[i]   = dists_src[i]
                cosmic_new[i]  = cosmic_src[i]
                dates_out.append(date_fmt)

                # Fix Dst if needed
                if cosmic_src[i, 1] == -15.0 or cosmic_src[i, 1] == 0.0:
                    if date_fmt in dst_lookup:
                        cosmic_new[i, 1] = float(dst_lookup[date_fmt])

                # Try to upgrade tensor from tensor/ folder
                if events_src[i] == 0 and date_str in new_noise_lookup:
                    # Build day tensor from processed/raw tensors
                    day_tensor = np.zeros((n_sta, 3, 128, 1440), dtype=np.float16)
                    active = 0
                    for station, fpath in new_noise_lookup[date_str].items():
                        if station in sta_idx:
                            try:
                                arr = np.load(fpath).astype(np.float16)
                                if arr.shape == (3, 128, 1440):
                                    day_tensor[sta_idx[station]] = arr
                                    active += 1
                            except Exception:
                                pass
                    if active > 0:
                        tds[i] = day_tensor
                        n_tensor_upgraded += 1
                    else:
                        # Keep original
                        tds[i] = tensors_src[i].astype(np.float16)
                else:
                    # Keep original tensor
                    tds[i] = tensors_src[i].astype(np.float16)

            # --- Add new noise days ---
            print(f"  Adding {n_add} new noise days...")
            for j, date_str in enumerate(new_dates_to_add):
                out_i = n_days + j
                date_fmt = date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8]

                # Build tensor
                day_tensor = np.zeros((n_sta, 3, 128, 1440), dtype=np.float16)
                active = 0
                for station, fpath in new_noise_lookup[date_str].items():
                    if station in sta_idx:
                        try:
                            arr = np.load(fpath).astype(np.float16)
                            if arr.shape == (3, 128, 1440):
                                day_tensor[sta_idx[station]] = arr
                                active += 1
                        except Exception:
                            pass
                tds[out_i] = day_tensor
                if active > 0:
                    n_tensor_new += 1

                # Labels (noise)
                events_new[out_i]  = 0
                mags_new[out_i]    = 0.0
                cosmic_new[out_i, 0] = 0.0  # Kp unknown
                cosmic_new[out_i, 1] = float(dst_lookup.get(date_fmt, 0.0))
                dates_out.append(date_fmt)

        print(f"\n  Tensor upgrades: {n_tensor_upgraded} noise days improved")
        print(f"  New tensor days: {n_tensor_new}/{n_add} with real data")

    return {
        'temp_file':   temp_file,
        'label_event': events_new,
        'label_mag':   mags_new,
        'label_azm':   azms_new,
        'label_dist':  dists_new,
        'cosmic':      cosmic_new,
        'dates':       np.array(dates_out, dtype='S10'),
        'stations':    all_stations,
        'n_days':      total_out,
    }


# ============================================================================
# STEP 4: SAVE AUGMENTED DATASET
# ============================================================================

def save_augmented(val_data):
    print("\n" + "=" * 70)
    print("STEP 4: SAVE AUGMENTED DATASET")
    print("=" * 70)

    with h5py.File(INPUT_FILE, 'r') as src, h5py.File(OUTPUT_FILE, 'w') as dst_f:
        # Copy global attrs
        for k, v in src.attrs.items():
            dst_f.attrs[k] = v
        dst_f.attrs['augmented']      = 'true'
        dst_f.attrs['augment_source'] = 'tensor/processed_tensors_v2 + tensor/raw_tensors_v2'

        # Copy TRAIN set unchanged
        print("\n  Copying TRAIN set (unchanged)...")
        src.copy('train', dst_f)
        tr_ev = src['train']['label_event'][:]
        print(f"    Train: {len(tr_ev)} days, {int(tr_ev.sum())} events")

        # Write augmented VAL set
        print("\n  Writing augmented VAL set...")
        grp = dst_f.create_group('val')
        with h5py.File(val_data['temp_file'], 'r') as tmp:
            tmp.copy('tensors', grp)
        grp.create_dataset('label_event',    data=val_data['label_event'])
        grp.create_dataset('label_mag',      data=val_data['label_mag'])
        grp.create_dataset('label_azm',      data=val_data['label_azm'])
        grp.create_dataset('label_dist',     data=val_data['label_dist'])
        grp.create_dataset('cosmic_features',data=val_data['cosmic'])
        grp.create_dataset('dates',          data=val_data['dates'])
        grp.attrs['num_stations'] = len(val_data['stations'])
        grp.attrs['stations']     = val_data['stations']
        grp.attrs['mw_threshold'] = MW_THRESHOLD

        n_ev = int(val_data['label_event'].sum())
        n_no = val_data['n_days'] - n_ev
        ratio = n_no / n_ev if n_ev > 0 else float('inf')
        print(f"    Val: {val_data['n_days']} days | {n_ev} events | {n_no} noise | ratio 1:{ratio:.1f}")

        os.remove(val_data['temp_file'])

        dst_f.attrs['created'] = datetime.now().isoformat()

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"\n  Saved: {OUTPUT_FILE}  ({size_mb:.1f} MB)")


# ============================================================================
# STEP 5: FINAL REPORT
# ============================================================================

def final_report():
    print("\n" + "=" * 70)
    print("STEP 5: FINAL COMPARISON REPORT")
    print("=" * 70)

    for fname, label in [(INPUT_FILE, 'BEFORE (V13 patched)'),
                         (OUTPUT_FILE, 'AFTER (V13 augmented)')]:
        if not os.path.exists(fname):
            continue
        size_mb = os.path.getsize(fname) / (1024**2)
        print(f"\n  {label}: {fname}  ({size_mb:.1f} MB)")
        with h5py.File(fname, 'r') as f:
            for grp_name in ['train', 'val']:
                if grp_name not in f:
                    continue
                g = f[grp_name]
                events    = g['label_event'][:]
                dates_raw = g['dates'][:]
                dates     = [d.decode() if isinstance(d, bytes) else d for d in dates_raw]
                n_ev = int(events.sum())
                n_no = len(events) - n_ev
                ratio = n_no / n_ev if n_ev > 0 else float('inf')

                # Tensor quality check
                tensors = g['tensors']
                sample_size = min(20, len(dates))
                nonzero = sum(1 for i in range(sample_size) if np.any(tensors[i] != 0))

                print(f"    [{grp_name.upper()}] {len(dates)} days | {n_ev} events | {n_no} noise | ratio 1:{ratio:.1f}")
                print(f"           Tensor quality (sample {sample_size}): {nonzero}/{sample_size} non-zero")
                print(f"           Date range: {min(dates)} to {max(dates)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("AUGMENT V13 DATASET WITH tensor/ FOLDER DATA")
    print("=" * 70)
    print(f"  Input : {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")

    # Load support data
    station_coords = load_stations()
    all_stations   = sorted(station_coords.keys())
    dst_lookup     = load_dst_lookup()
    print(f"\n  Stations: {len(station_coords)}")
    print(f"  Dst records: {len(dst_lookup)}")

    # Step 1: Inventory new noise
    new_noise = inventory_new_noise()

    # Step 2: Check existing val
    existing_noise_dates = check_existing_val()

    # Step 3: Build augmented val
    val_data = build_augmented_val(
        new_noise, existing_noise_dates,
        station_coords, all_stations, dst_lookup
    )

    # Step 4: Save
    save_augmented(val_data)

    # Step 5: Report
    final_report()

    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
