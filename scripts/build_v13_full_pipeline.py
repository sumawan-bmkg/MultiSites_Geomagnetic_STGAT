"""
V13 Full Pipeline: Threshold Calibration + Full Dataset Generation
"""
import os, h5py, numpy as np, pandas as pd, warnings
from datetime import datetime, date, timezone
from math import radians, sin, cos, sqrt, atan2, degrees
warnings.filterwarnings('ignore')

MW_THRESHOLD = 5.0
TRAIN_END    = date(2023, 12, 31)
VAL_START    = date(2024,  1,  1)
VAL_END      = date(2025,  3, 31)
BLIND_START  = date(2026,  1,  1)
TARGET_RATIO = 4

V12_BLINDTEST  = 'dataset_v12_blindtest.h5'
V13_BLINDTEST  = 'dataset_v13_blindtest_M5.h5'
V13_TRAIN_VAL  = 'dataset_v13_train_val_M5.h5'
STATION_FILE   = 'intial/lokasi_stasiun.csv'
RAW_TENSOR_DIR = 'tensor/raw_tensors_v2'
SOURCE_HDF5    = ['scalogram_v8_true_negatives.h5','scalogram_v8_hard_negatives.h5','scalogram_v3_cosmic_final.h5']

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
    az = degrees(atan2(x, y))
    return (az + 360) % 360

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

def load_eq_catalog():
    dfs = []
    for fname in ['intial/EQ1.2026.csv','intial/EQ2.2026.csv','intial/EQ3.2026.csv','intial/EQ4.2026.csv']:
        if os.path.exists(fname):
            dfs.append(pd.read_csv(fname))
    if not dfs:
        return {}
    all_eq = pd.concat(dfs, ignore_index=True)
    all_eq['datetime'] = pd.to_datetime(all_eq['Date time'], utc=True)
    all_eq['date'] = all_eq['datetime'].dt.date
    all_eq = all_eq[all_eq['Magnitude'] >= MW_THRESHOLD].copy()
    eq_by_date = {}
    for _, row in all_eq.iterrows():
        date_str = str(row['date']).replace('-', '')
        if date_str not in eq_by_date or row['Magnitude'] > eq_by_date[date_str]['mag']:
            eq_by_date[date_str] = {'mag': float(row['Magnitude']), 'lat': float(row['Latitude']), 'lon': float(row['Longitude'])}
    return eq_by_date

def load_hist_eq_catalog():
    eq_by_date = {}
    for fname in ['intial/earthquake_catalog_2018_2025_merged_robust.csv']:
        if not os.path.exists(fname):
            continue
        df = pd.read_csv(fname)
        print('  Catalog columns:', list(df.columns)[:8])
        print('  Catalog shape:', df.shape)
        print('  Sample:', df.head(2).to_string())
        break
    return eq_by_date

V11_SOURCE = 'dataset_v11_patched.h5'

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
    az = degrees(atan2(x, y))
    return (az + 360) % 360

def load_stations():
    """Load 24 station coordinates from lokasi_stasiun.csv."""
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

def load_eq_catalog_2026():
    """Load EQ catalog 2026 for blindtest label lookup."""
    dfs = []
    for fname in ['intial/EQ1.2026.csv', 'intial/EQ2.2026.csv',
                  'intial/EQ3.2026.csv', 'intial/EQ4.2026.csv']:
        if os.path.exists(fname):
            dfs.append(pd.read_csv(fname))
    if not dfs:
        return {}
    all_eq = pd.concat(dfs, ignore_index=True)
    all_eq['dt'] = pd.to_datetime(all_eq['Date time'], utc=True, errors='coerce')
    all_eq = all_eq.dropna(subset=['dt', 'Latitude', 'Longitude', 'Magnitude'])
    all_eq = all_eq[all_eq['Magnitude'] >= MW_THRESHOLD].copy()
    eq_by_date = {}
    for _, row in all_eq.iterrows():
        date_str = row['dt'].strftime('%Y%m%d')
        if date_str not in eq_by_date or row['Magnitude'] > eq_by_date[date_str]['mag']:
            eq_by_date[date_str] = {
                'mag': float(row['Magnitude']),
                'lat': float(row['Latitude']),
                'lon': float(row['Longitude'])
            }
    return eq_by_date

def load_eq_catalog_historical():
    """Load historical EQ catalog for train/val label lookup."""
    eq_by_date = {}
    # Try merged robust catalog
    for fname in ['intial/earthquake_catalog_2018_2025_merged_robust.csv',
                  'intial/earthquake_catalog_2018_2025_merged.csv']:
        if not os.path.exists(fname):
            continue
        df = pd.read_csv(fname)
        # Try 'Date time' column
        for col in ['Date time', 'datetime']:
            if col in df.columns:
                df['dt'] = pd.to_datetime(df[col], utc=True, errors='coerce')
                break
        if 'dt' not in df.columns:
            continue
        df = df.dropna(subset=['dt', 'Latitude', 'Longitude', 'Magnitude'])
        df = df[df['Magnitude'] >= MW_THRESHOLD].copy()
        for _, row in df.iterrows():
            date_str = row['dt'].strftime('%Y%m%d')
            if date_str not in eq_by_date or row['Magnitude'] > eq_by_date[date_str]['mag']:
                eq_by_date[date_str] = {
                    'mag': float(row['Magnitude']),
                    'lat': float(row['Latitude']),
                    'lon': float(row['Longitude'])
                }
    # Also load EQ Repository files (Sep-Dec 2025)
    for fname in ['intial/EQ Repository  september Requested Data.csv',
                  'intial/EQ Repository oktober Requested Data.csv',
                  'intial/EQ Repository november  Requested Data.csv',
                  'intial/EQ Repository desember Requested Data.csv']:
        if not os.path.exists(fname):
            continue
        df = pd.read_csv(fname)
        df['dt'] = pd.to_datetime(df['Date time'], utc=True, errors='coerce')
        df = df.dropna(subset=['dt', 'Latitude', 'Longitude', 'Magnitude'])
        df = df[df['Magnitude'] >= MW_THRESHOLD].copy()
        for _, row in df.iterrows():
            date_str = row['dt'].strftime('%Y%m%d')
            if date_str not in eq_by_date or row['Magnitude'] > eq_by_date[date_str]['mag']:
                eq_by_date[date_str] = {
                    'mag': float(row['Magnitude']),
                    'lat': float(row['Latitude']),
                    'lon': float(row['Longitude'])
                }
    return eq_by_date

def load_kp_dst_for_date(date_str):
    """Load Kp and Dst for a given date from blindtest/kp_index/ or return defaults."""
    kp_val, dst_val = 0.0, 0.0
    kp_dir = 'blindtest/kp_index'
    if not os.path.exists(kp_dir):
        return kp_val, dst_val
    # DST file
    dst_file = os.path.join(kp_dir, f'dst_index_{date_str}.csv')
    if os.path.exists(dst_file):
        try:
            df = pd.read_csv(dst_file)
            if 'dst_value' in df.columns:
                dst_val = float(df['dst_value'].mean())
        except Exception:
            pass
    # KP file by timestamp
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        ts = int(dt.timestamp())
        kp_file = os.path.join(kp_dir, f'kp_index_{ts}.csv')
        if os.path.exists(kp_file):
            df = pd.read_csv(kp_file)
            if 'kp_value' in df.columns:
                kp_val = float(df['kp_value'].max())
    except Exception:
        pass
    return kp_val, dst_val


# ============================================================================
# TAHAP 1: KALIBRASI BLINDTEST V12 -> V13 (Mw >= 5.0)
# ============================================================================

def tahap1_calibrate_blindtest(station_coords, all_stations):
    print("=" * 70)
    print("TAHAP 1: KALIBRASI BLINDTEST V12 -> V13 (Mw >= 5.0)")
    print("=" * 70)

    eq_catalog = load_eq_catalog_2026()
    n_sta = len(all_stations)
    sta_idx = {s: i for i, s in enumerate(all_stations)}

    with h5py.File(V12_BLINDTEST, 'r') as src:
        g = src['test']
        tensors_src   = g['tensors']
        events_src    = g['label_event'][:]
        mags_src      = g['label_mag'][:]
        azms_src      = g['label_azm'][:]
        dists_src     = g['label_dist'][:] if 'label_dist' in g else np.zeros((len(events_src), tensors_src.shape[1]), dtype=np.float32)
        cosmic_src    = g['cosmic_features'][:]
        dates_src     = g['dates'][:]
        src_stations  = list(g.attrs.get('stations', []))
        if src_stations and isinstance(src_stations[0], bytes):
            src_stations = [s.decode() for s in src_stations]

        n_days = tensors_src.shape[0]
        n_src_sta = tensors_src.shape[1]

        # Decode dates
        dates_str = [d.decode() if isinstance(d, bytes) else d for d in dates_src]

        # Apply Mw >= 5.0 filter + expand to 24 stations
        events_new = np.zeros(n_days, dtype=np.int8)
        mags_new   = np.zeros(n_days, dtype=np.float32)
        azms_new   = np.zeros((n_days, n_sta), dtype=np.float32)
        dists_new  = np.zeros((n_days, n_sta), dtype=np.float32)

        n_kept = 0
        n_downgraded = 0

        for i in range(n_days):
            date_fmt = dates_str[i]  # YYYY-MM-DD
            date_key = date_fmt.replace('-', '')

            if events_src[i] == 1 and mags_src[i] >= MW_THRESHOLD:
                events_new[i] = 1
                mags_new[i]   = mags_src[i]
                n_kept += 1

                # Compute azimuth and distance per station using catalog
                eq_info = eq_catalog.get(date_key)
                if eq_info:
                    eq_lat, eq_lon = eq_info['lat'], eq_info['lon']
                    for station, (sta_lat, sta_lon) in station_coords.items():
                        if station in sta_idx:
                            s_i = sta_idx[station]
                            azms_new[i, s_i]  = azimuth_calc(sta_lat, sta_lon, eq_lat, eq_lon)
                            dists_new[i, s_i] = haversine(sta_lat, sta_lon, eq_lat, eq_lon)
                else:
                    # Remap from source azm/dist
                    for s, s_i in sta_idx.items():
                        if s in src_stations:
                            old_i = src_stations.index(s)
                            if old_i < azms_src.shape[1]:
                                azms_new[i, s_i]  = azms_src[i, old_i]
                                dists_new[i, s_i] = dists_src[i, old_i]
            else:
                if events_src[i] == 1:
                    n_downgraded += 1
                # noise: all zeros

        n_ev_new = int(events_new.sum())
        n_no_new = n_days - n_ev_new

        print(f"  Original: {int(events_src.sum())} events / {n_days - int(events_src.sum())} noise")
        print(f"  Downgraded (Mw < {MW_THRESHOLD}): {n_downgraded}")
        print(f"  After calibration: {n_ev_new} events / {n_no_new} noise")
        ratio = n_no_new / n_ev_new if n_ev_new > 0 else float('inf')
        print(f"  Ratio: 1:{ratio:.1f}")

        # Save V13 blindtest with 24 stations
        with h5py.File(V13_BLINDTEST, 'w') as dst:
            test_grp = dst.create_group('test')
            tensors_ds = test_grp.create_dataset(
                'tensors', shape=(n_days, n_sta, 3, 128, 1440),
                dtype=np.float16, compression='gzip',
                chunks=(1, n_sta, 3, 128, 1440)
            )
            print(f"  Copying and expanding tensors to {n_sta} stations...")
            for i in range(n_days):
                day_tensor = np.zeros((n_sta, 3, 128, 1440), dtype=np.float16)
                src_tensor = tensors_src[i]  # (n_src_sta, 3, 128, 1440)
                for s in src_stations:
                    if s in sta_idx:
                        old_i = src_stations.index(s)
                        new_i = sta_idx[s]
                        if old_i < src_tensor.shape[0]:
                            day_tensor[new_i] = src_tensor[old_i]
                tensors_ds[i] = day_tensor

            test_grp.create_dataset('label_event',    data=events_new)
            test_grp.create_dataset('label_mag',      data=mags_new)
            test_grp.create_dataset('label_azm',      data=azms_new)
            test_grp.create_dataset('label_dist',     data=dists_new)
            test_grp.create_dataset('cosmic_features',data=cosmic_src)
            test_grp.create_dataset('dates',          data=dates_src)
            test_grp.attrs['num_stations'] = n_sta
            test_grp.attrs['stations']     = all_stations
            test_grp.attrs['mw_threshold'] = MW_THRESHOLD
            test_grp.attrs['n_events']     = n_ev_new
            test_grp.attrs['n_noise']      = n_no_new

            dst.attrs['created']       = datetime.now().isoformat()
            dst.attrs['version']       = 'v13'
            dst.attrs['mw_threshold']  = MW_THRESHOLD
            dst.attrs['source']        = V12_BLINDTEST

    size_mb = os.path.getsize(V13_BLINDTEST) / (1024**2)
    print(f"  Saved: {V13_BLINDTEST} ({size_mb:.1f} MB)")
    print(f"  [RESULT] Blindtest: {n_ev_new} events ({n_ev_new/n_days*100:.1f}%) / {n_no_new} noise ({n_no_new/n_days*100:.1f}%)")
    return n_ev_new, n_no_new


# ============================================================================
# TAHAP 2-5: BUILD TRAIN & VAL FROM V11 + EXPAND TO 24 STATIONS
# ============================================================================

def build_train_val(station_coords, all_stations):
    print("\n" + "=" * 70)
    print("TAHAP 2-5: BUILD TRAIN & VAL (V11 -> 24 stations, Mw >= 5.0)")
    print("=" * 70)

    n_sta = len(all_stations)
    sta_idx = {s: i for i, s in enumerate(all_stations)}

    # Load historical EQ catalog for azm/dist computation
    eq_hist = load_eq_catalog_historical()
    print(f"  Historical EQ catalog (Mw>={MW_THRESHOLD}): {len(eq_hist)} event dates")

    with h5py.File(V11_SOURCE, 'r') as src_f:
        with h5py.File(V13_TRAIN_VAL, 'w') as out_f:

            for split_name in ['train', 'val']:
                if split_name not in src_f:
                    print(f"  SKIP {split_name}: not in source")
                    continue

                g = src_f[split_name]
                tensors_src   = g['tensors']
                events_src    = g['label_event'][:]
                mags_src      = g['label_mag'][:]
                azms_src      = g['label_azm'][:]
                cosmic_src    = g['cosmic_features'][:]
                dates_src_raw = g['dates'][:]
                src_stations  = list(g.attrs.get('stations', []))
                if src_stations and isinstance(src_stations[0], bytes):
                    src_stations = [s.decode() for s in src_stations]

                n_days = tensors_src.shape[0]
                dates_str = [d.decode() if isinstance(d, bytes) else d for d in dates_src_raw]
                dates_key = [d.replace('-', '') for d in dates_str]

                print(f"\n  Processing {split_name.upper()}:")
                print(f"    Source: {n_days} days, {len(src_stations)} stations")
                print(f"    Date range: {min(dates_str)} to {max(dates_str)}")

                # Apply Mw >= 5.0 filter
                events_new = np.zeros(n_days, dtype=np.int8)
                mags_new   = np.zeros(n_days, dtype=np.float32)
                azms_new   = np.zeros((n_days, n_sta), dtype=np.float32)
                dists_new  = np.zeros((n_days, n_sta), dtype=np.float32)
                cosmic_new = np.zeros((n_days, 2), dtype=np.float32)

                n_kept = 0
                n_downgraded = 0

                for i in range(n_days):
                    date_key = dates_key[i]

                    # Cosmic features
                    cosmic_new[i, 0] = cosmic_src[i, 0]  # Kp
                    cosmic_new[i, 1] = cosmic_src[i, 1]  # Dst
                    # Fix suspicious Dst (-15.0 constant)
                    if cosmic_src[i, 1] == -15.0:
                        kp_live, dst_live = load_kp_dst_for_date(date_key)
                        if dst_live != 0.0:
                            cosmic_new[i, 1] = dst_live

                    if events_src[i] == 1 and mags_src[i] >= MW_THRESHOLD:
                        events_new[i] = 1
                        mags_new[i]   = mags_src[i]
                        n_kept += 1

                        # Compute azm/dist from catalog
                        eq_info = eq_hist.get(date_key)
                        if eq_info:
                            eq_lat, eq_lon = eq_info['lat'], eq_info['lon']
                            for station, (sta_lat, sta_lon) in station_coords.items():
                                if station in sta_idx:
                                    s_i = sta_idx[station]
                                    azms_new[i, s_i]  = azimuth_calc(sta_lat, sta_lon, eq_lat, eq_lon)
                                    dists_new[i, s_i] = haversine(sta_lat, sta_lon, eq_lat, eq_lon)
                        else:
                            # Remap from source azm (per-station scalar)
                            for s in src_stations:
                                if s in sta_idx:
                                    old_i = src_stations.index(s)
                                    new_i = sta_idx[s]
                                    if old_i < azms_src.shape[1]:
                                        azms_new[i, new_i] = azms_src[i, old_i]
                    else:
                        if events_src[i] == 1:
                            n_downgraded += 1

                n_ev = int(events_new.sum())
                n_no = n_days - n_ev
                print(f"    Events (Mw>={MW_THRESHOLD}): {n_ev} | Noise: {n_no} | Downgraded: {n_downgraded}")

                # Data balancing: undersample noise if ratio > TARGET_RATIO
                keep_indices = list(range(n_days))
                if n_ev > 0 and n_no > n_ev * TARGET_RATIO:
                    event_idx = [i for i in range(n_days) if events_new[i] == 1]
                    noise_idx = [i for i in range(n_days) if events_new[i] == 0]
                    target_noise = n_ev * TARGET_RATIO

                    # Sample noise evenly across months
                    noise_dates = [dates_str[i] for i in noise_idx]
                    noise_df = pd.DataFrame({'idx': noise_idx, 'date': noise_dates})
                    noise_df['month'] = noise_df['date'].str[:7]
                    n_months = noise_df['month'].nunique()
                    spm = max(1, int(target_noise / n_months))
                    sampled = (noise_df.groupby('month')
                               .apply(lambda x: x.sample(min(spm, len(x)), random_state=42))
                               .reset_index(drop=True)['idx'].tolist())
                    keep_indices = sorted(event_idx + sampled)
                    n_no = len(sampled)
                    print(f"    After balancing: {len(keep_indices)} days, {n_ev} events, {n_no} noise (1:{n_no/n_ev:.1f})")

                n_out = len(keep_indices)

                # Write output group
                grp = out_f.create_group(split_name)
                tensors_ds = grp.create_dataset(
                    'tensors', shape=(n_out, n_sta, 3, 128, 1440),
                    dtype=np.float16, compression='gzip',
                    chunks=(1, n_sta, 3, 128, 1440)
                )

                active_counts = []
                for out_idx, src_idx_i in enumerate(keep_indices):
                    if out_idx % 50 == 0:
                        print(f"    Writing day {out_idx+1}/{n_out}...")
                    day_tensor = np.zeros((n_sta, 3, 128, 1440), dtype=np.float16)
                    src_tensor = tensors_src[src_idx_i]  # (n_src_sta, 3, 128, 1440)
                    active = 0
                    for s in src_stations:
                        if s in sta_idx:
                            old_i = src_stations.index(s)
                            new_i = sta_idx[s]
                            if old_i < src_tensor.shape[0]:
                                t = src_tensor[old_i]
                                if np.any(t != 0):
                                    day_tensor[new_i] = t
                                    active += 1
                    tensors_ds[out_idx] = day_tensor
                    active_counts.append(active)

                avg_active = np.mean(active_counts) if active_counts else 0

                # Save labels
                sel_events = events_new[keep_indices]
                sel_mags   = mags_new[keep_indices]
                sel_azms   = azms_new[keep_indices]
                sel_dists  = dists_new[keep_indices]
                sel_cosmic = cosmic_new[keep_indices]
                sel_dates  = np.array([dates_src_raw[i] for i in keep_indices])

                grp.create_dataset('label_event',    data=sel_events)
                grp.create_dataset('label_mag',      data=sel_mags)
                grp.create_dataset('label_azm',      data=sel_azms)
                grp.create_dataset('label_dist',     data=sel_dists)
                grp.create_dataset('cosmic_features',data=sel_cosmic)
                grp.create_dataset('dates',          data=sel_dates)
                grp.attrs['num_stations']        = n_sta
                grp.attrs['stations']            = all_stations
                grp.attrs['mw_threshold']        = MW_THRESHOLD
                grp.attrs['avg_active_stations'] = avg_active

                n_ev_final = int(sel_events.sum())
                n_no_final = n_out - n_ev_final
                ratio_final = n_no_final / n_ev_final if n_ev_final > 0 else float('inf')
                print(f"    [RESULT] {split_name.upper()}: {n_out} days | {n_ev_final} events ({n_ev_final/n_out*100:.0f}%) | {n_no_final} noise | ratio 1:{ratio_final:.1f}")
                print(f"    Avg active stations/day: {avg_active:.1f}/{n_sta}")

            out_f.attrs['created']      = datetime.now().isoformat()
            out_f.attrs['version']      = 'v13'
            out_f.attrs['mw_threshold'] = MW_THRESHOLD
            out_f.attrs['train_end']    = str(TRAIN_END)
            out_f.attrs['val_start']    = str(VAL_START)
            out_f.attrs['val_end']      = str(VAL_END)

    size_mb = os.path.getsize(V13_TRAIN_VAL) / (1024**2)
    print(f"\n  Saved: {V13_TRAIN_VAL} ({size_mb:.1f} MB)")


# ============================================================================
# FINAL REPORT
# ============================================================================

def final_report():
    print("\n" + "=" * 70)
    print("FINAL REPORT - V13 DATASET SUITE")
    print("=" * 70)

    for fname, label in [(V13_BLINDTEST, 'BLINDTEST'), (V13_TRAIN_VAL, 'TRAIN/VAL')]:
        if not os.path.exists(fname):
            continue
        size_mb = os.path.getsize(fname) / (1024**2)
        print(f"\n  {label}: {fname}  ({size_mb:.1f} MB)")
        with h5py.File(fname, 'r') as f:
            for grp_name in f.keys():
                g = f[grp_name]
                events    = g['label_event'][:]
                mags      = g['label_mag'][:]
                dates_raw = g['dates'][:]
                dates = [d.decode() if isinstance(d, bytes) else d for d in dates_raw]
                stations = list(g.attrs.get('stations', []))
                if stations and isinstance(stations[0], bytes):
                    stations = [s.decode() for s in stations]
                n_ev = int(events.sum())
                n_no = len(events) - n_ev
                ratio = n_no / n_ev if n_ev > 0 else float('inf')
                ev_mags = mags[events == 1]
                mag_str = f"{ev_mags.min():.2f}-{ev_mags.max():.2f}" if len(ev_mags) > 0 else "N/A"
                avg_active = g.attrs.get('avg_active_stations', 'N/A')
                cosmic = g['cosmic_features'][:]

                print(f"    [{grp_name.upper()}]")
                print(f"      Days      : {len(dates)}")
                print(f"      Events    : {n_ev} ({n_ev/len(dates)*100:.0f}%)")
                print(f"      Noise     : {n_no} ({n_no/len(dates)*100:.0f}%)")
                print(f"      Ratio     : 1:{ratio:.1f}")
                print(f"      Mag range : {mag_str} Mw")
                print(f"      Stations  : {len(stations)}")
                print(f"      Avg active: {avg_active}")
                print(f"      Date range: {min(dates)} to {max(dates)}")
                print(f"      Kp range  : {cosmic[:,0].min():.2f} to {cosmic[:,0].max():.2f}")
                print(f"      Dst range : {cosmic[:,1].min():.2f} to {cosmic[:,1].max():.2f}")
                # NaN check
                sample = g['tensors'][:3]
                nan_count = int(np.isnan(sample).sum())
                print(f"      NaN check : {nan_count} NaN in first 3 days")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("V13 FULL PIPELINE: THRESHOLD CALIBRATION + DATASET GENERATION")
    print(f"  Mw threshold : >= {MW_THRESHOLD}")
    print(f"  Train period : <= {TRAIN_END}")
    print(f"  Val period   : {VAL_START} to {VAL_END}")
    print(f"  Target ratio : 1:{TARGET_RATIO}")
    print("=" * 70)

    # Load station coordinates (24 stations)
    station_coords = load_stations()
    all_stations = sorted(station_coords.keys())
    print(f"\nStation coordinates: {len(station_coords)} stations")
    print(f"  {all_stations}")

    # TAHAP 1: Calibrate blindtest
    tahap1_calibrate_blindtest(station_coords, all_stations)

    # TAHAP 2-5: Build train/val
    build_train_val(station_coords, all_stations)

    # Final report
    final_report()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Output 1: {V13_BLINDTEST}")
    print(f"  Output 2: {V13_TRAIN_VAL}")


if __name__ == '__main__':
    main()
