import os, h5py, numpy as np, pandas as pd, warnings
from datetime import datetime, date, timezone
from math import radians, sin, cos, sqrt, atan2, degrees
warnings.filterwarnings("ignore")

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
V11_SOURCE     = 'dataset_v11_patched.h5'
RAW_TENSOR_DIR = 'tensor/raw_tensors_v2'
SOURCE_HDF5    = ['scalogram_v8_true_negatives.h5','scalogram_v8_hard_negatives.h5','scalogram_v3_cosmic_final.h5']

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
    """Load all available EQ catalogs (2026 + historical)."""
    eq_by_date = {}
    # 2026 catalogs
    for fname in ['intial/EQ1.2026.csv', 'intial/EQ2.2026.csv',
                  'intial/EQ3.2026.csv', 'intial/EQ4.2026.csv']:
        if not os.path.exists(fname):
            continue
        df = pd.read_csv(fname)
        df['dt'] = pd.to_datetime(df['Date time'], utc=True, errors='coerce')
        df = df.dropna(subset=['dt', 'Latitude', 'Longitude', 'Magnitude'])
        df = df[df['Magnitude'] >= MW_THRESHOLD]
        for _, row in df.iterrows():
            ds = row['dt'].strftime('%Y%m%d')
            if ds not in eq_by_date or row['Magnitude'] > eq_by_date[ds]['mag']:
                eq_by_date[ds] = {'mag': float(row['Magnitude']),
                                  'lat': float(row['Latitude']),
                                  'lon': float(row['Longitude'])}
    # Historical catalogs
    for fname in ['intial/earthquake_catalog_2018_2025_merged_robust.csv',
                  'intial/earthquake_catalog_2018_2025_merged.csv',
                  'intial/EQ Repository  september Requested Data.csv',
                  'intial/EQ Repository oktober Requested Data.csv',
                  'intial/EQ Repository november  Requested Data.csv',
                  'intial/EQ Repository desember Requested Data.csv']:
        if not os.path.exists(fname):
            continue
        df = pd.read_csv(fname)
        for col in ['Date time', 'datetime']:
            if col in df.columns:
                df['dt'] = pd.to_datetime(df[col], utc=True, errors='coerce')
                break
        if 'dt' not in df.columns:
            continue
        df = df.dropna(subset=['dt', 'Latitude', 'Longitude', 'Magnitude'])
        df = df[df['Magnitude'] >= MW_THRESHOLD]
        for _, row in df.iterrows():
            ds = row['dt'].strftime('%Y%m%d')
            if ds not in eq_by_date or row['Magnitude'] > eq_by_date[ds]['mag']:
                eq_by_date[ds] = {'mag': float(row['Magnitude']),
                                  'lat': float(row['Latitude']),
                                  'lon': float(row['Longitude'])}
    return eq_by_date

def build_source_lookup():
    """Build (date, station) -> (fname, grp, idx) lookup from source HDF5 files."""
    lookup = {}
    for fname in SOURCE_HDF5:
        if not os.path.exists(fname):
            continue
        with h5py.File(fname, 'r') as f:
            for grp_name in f.keys():
                g = f[grp_name]
                if 'meta' not in g:
                    continue
                meta = g['meta'][:]
                for i, m in enumerate(meta):
                    if isinstance(m, bytes):
                        m = m.decode()
                    parts = str(m).split('_')
                    if len(parts) >= 3:
                        station = parts[1]
                        date_str = parts[2].replace('.npy', '')
                        if len(date_str) == 8 and date_str.isdigit():
                            key = (date_str, station)
                            if key not in lookup:
                                lookup[key] = (fname, grp_name, i)
    return lookup

def build_raw_tensor_lookup():
    """Build (date, station) -> filepath lookup from raw_tensors_v2."""
    lookup = {}
    if not os.path.exists(RAW_TENSOR_DIR):
        return lookup
    for fname in os.listdir(RAW_TENSOR_DIR):
        if not fname.endswith('.npy') or fname.startswith('desktop'):
            continue
        parts = fname.replace('raw_tensor_', '').replace('.npy', '').split('_')
        if len(parts) == 2:
            station, ts_str = parts
            try:
                ts = int(ts_str)
                dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
                date_str = dt.strftime('%Y%m%d')
                key = (date_str, station)
                if key not in lookup:
                    lookup[key] = os.path.join(RAW_TENSOR_DIR, fname)
            except Exception:
                pass
    return lookup

def get_tensor_for_station(station, date_str, src_lookup, rt_lookup, open_files):
    """
    Try to get a real tensor for (station, date).
    Priority: raw_tensors_v2 > source HDF5 (same date) > source HDF5 (any date, same station)
    Returns (3, 128, 1440) float16 or None.
    """
    # 1. Try raw_tensors_v2 exact date
    key = (date_str, station)
    if key in rt_lookup:
        try:
            arr = np.load(rt_lookup[key]).astype(np.float16)
            if arr.shape == (3, 128, 1440):
                return arr
        except Exception:
            pass

    # 2. Try source HDF5 exact date
    if key in src_lookup:
        fname, grp, idx = src_lookup[key]
        try:
            f = open_files.get(fname)
            if f is not None:
                tensor = f[grp]['tensors'][idx].astype(np.float16)
                if tensor.shape == (3, 128, 1440):
                    return tensor
        except Exception:
            pass

    # 3. Try source HDF5 any date for same station (proxy)
    station_keys = [k for k in src_lookup.keys() if k[1] == station]
    if station_keys:
        # Use the most recent available date as proxy
        station_keys.sort(key=lambda k: k[0], reverse=True)
        for proxy_key in station_keys[:3]:
            fname, grp, idx = src_lookup[proxy_key]
            try:
                f = open_files.get(fname)
                if f is not None:
                    tensor = f[grp]['tensors'][idx].astype(np.float16)
                    if tensor.shape == (3, 128, 1440):
                        return tensor
            except Exception:
                pass

    return None


# ============================================================================
# TAHAP 1: KALIBRASI BLINDTEST V12 -> V13
# ============================================================================

def tahap1_calibrate_blindtest(station_coords, all_stations):
    print("=" * 70)
    print("TAHAP 1: KALIBRASI BLINDTEST V12 -> V13 (Mw >= 5.0)")
    print("=" * 70)

    n_sta = len(all_stations)
    sta_idx = {s: i for i, s in enumerate(all_stations)}
    eq_catalog = load_eq_catalog()

    with h5py.File(V12_BLINDTEST, 'r') as src:
        g = src['test']
        tensors_src  = g['tensors']
        events_src   = g['label_event'][:]
        mags_src     = g['label_mag'][:]
        azms_src     = g['label_azm'][:]
        dists_src    = g['label_dist'][:] if 'label_dist' in g else np.zeros((len(events_src), tensors_src.shape[1]), dtype=np.float32)
        cosmic_src   = g['cosmic_features'][:]
        dates_src    = g['dates'][:]
        src_stations = list(g.attrs.get('stations', []))
        if src_stations and isinstance(src_stations[0], bytes):
            src_stations = [s.decode() for s in src_stations]

        n_days   = tensors_src.shape[0]
        n_src_st = tensors_src.shape[1]
        dates_str = [d.decode() if isinstance(d, bytes) else d for d in dates_src]

        events_new = np.zeros(n_days, dtype=np.int8)
        mags_new   = np.zeros(n_days, dtype=np.float32)
        azms_new   = np.zeros((n_days, n_sta), dtype=np.float32)
        dists_new  = np.zeros((n_days, n_sta), dtype=np.float32)

        n_kept = 0
        n_downgraded = 0

        for i in range(n_days):
            date_key = dates_str[i].replace('-', '')
            if events_src[i] == 1 and mags_src[i] >= MW_THRESHOLD:
                events_new[i] = 1
                mags_new[i]   = mags_src[i]
                n_kept += 1
                eq_info = eq_catalog.get(date_key)
                if eq_info:
                    eq_lat, eq_lon = eq_info['lat'], eq_info['lon']
                    for station, (slat, slon) in station_coords.items():
                        if station in sta_idx:
                            si = sta_idx[station]
                            azms_new[i, si]  = azimuth_calc(slat, slon, eq_lat, eq_lon)
                            dists_new[i, si] = haversine(slat, slon, eq_lat, eq_lon)
                else:
                    for s in src_stations:
                        if s in sta_idx:
                            oi = src_stations.index(s)
                            ni = sta_idx[s]
                            if oi < azms_src.shape[1]:
                                azms_new[i, ni]  = azms_src[i, oi]
                                dists_new[i, ni] = dists_src[i, oi]
            elif events_src[i] == 1:
                n_downgraded += 1

        n_ev_new = int(events_new.sum())
        n_no_new = n_days - n_ev_new
        print(f"  Original : {int(events_src.sum())} events / {n_days - int(events_src.sum())} noise")
        print(f"  Downgraded (Mw < {MW_THRESHOLD}): {n_downgraded}")
        print(f"  After    : {n_ev_new} events / {n_no_new} noise  (ratio 1:{n_no_new/n_ev_new:.1f})")

        with h5py.File(V13_BLINDTEST, 'w') as dst:
            tg = dst.create_group('test')
            tds = tg.create_dataset('tensors',
                shape=(n_days, n_sta, 3, 128, 1440), dtype=np.float16,
                compression='gzip', chunks=(1, n_sta, 3, 128, 1440))
            print(f"  Copying tensors ({n_days} days, expanding to {n_sta} stations)...")
            for i in range(n_days):
                day = np.zeros((n_sta, 3, 128, 1440), dtype=np.float16)
                src_t = tensors_src[i]
                for s in src_stations:
                    if s in sta_idx:
                        oi = src_stations.index(s)
                        ni = sta_idx[s]
                        if oi < src_t.shape[0]:
                            day[ni] = src_t[oi]
                tds[i] = day
            tg.create_dataset('label_event',    data=events_new)
            tg.create_dataset('label_mag',      data=mags_new)
            tg.create_dataset('label_azm',      data=azms_new)
            tg.create_dataset('label_dist',     data=dists_new)
            tg.create_dataset('cosmic_features',data=cosmic_src)
            tg.create_dataset('dates',          data=dates_src)
            tg.attrs['num_stations'] = n_sta
            tg.attrs['stations']     = all_stations
            tg.attrs['mw_threshold'] = MW_THRESHOLD
            tg.attrs['n_events']     = n_ev_new
            tg.attrs['n_noise']      = n_no_new
            dst.attrs['created']     = datetime.now().isoformat()
            dst.attrs['version']     = 'v13'
            dst.attrs['mw_threshold']= MW_THRESHOLD

    size_mb = os.path.getsize(V13_BLINDTEST) / (1024**2)
    print(f"  Saved: {V13_BLINDTEST}  ({size_mb:.1f} MB)")
    print(f"  [RESULT] Blindtest: {n_ev_new} events ({n_ev_new/n_days*100:.1f}%) / {n_no_new} noise ({n_no_new/n_days*100:.1f}%)")
    return n_ev_new, n_no_new


# ============================================================================
# TAHAP 2-5: BUILD TRAIN & VAL (24 stations, real tensors where available)
# ============================================================================

def build_split(split_name, station_coords, all_stations,
                src_lookup, rt_lookup, open_files, eq_catalog):
    """
    Build one split (train or val) from V11 labels + real tensors.
    Returns dict with temp_file path and label arrays.
    """
    n_sta   = len(all_stations)
    sta_idx = {s: i for i, s in enumerate(all_stations)}

    with h5py.File(V11_SOURCE, 'r') as src:
        g = src[split_name]
        events_src   = g['label_event'][:]
        mags_src     = g['label_mag'][:]
        azms_src     = g['label_azm'][:]
        cosmic_src   = g['cosmic_features'][:]
        dates_src    = g['dates'][:]
        src_stations = list(g.attrs.get('stations', []))
        if src_stations and isinstance(src_stations[0], bytes):
            src_stations = [s.decode() for s in src_stations]

    n_days    = len(events_src)
    dates_str = [d.decode() if isinstance(d, bytes) else d for d in dates_src]
    dates_key = [d.replace('-', '') for d in dates_str]

    # Apply Mw >= 5.0 filter and compute dist labels
    events_new = np.zeros(n_days, dtype=np.int8)
    mags_new   = np.zeros(n_days, dtype=np.float32)
    azms_new   = np.zeros((n_days, n_sta), dtype=np.float32)
    dists_new  = np.zeros((n_days, n_sta), dtype=np.float32)
    cosmic_new = cosmic_src.copy().astype(np.float32)

    n_kept = 0
    n_downgraded = 0

    for i in range(n_days):
        dk = dates_key[i]
        # Fix suspicious Dst
        if cosmic_src[i, 1] == -15.0:
            kp_dir = 'blindtest/kp_index'
            dst_file = os.path.join(kp_dir, f'dst_index_{dk}.csv')
            if os.path.exists(dst_file):
                try:
                    df_dst = pd.read_csv(dst_file)
                    if 'dst_value' in df_dst.columns:
                        cosmic_new[i, 1] = float(df_dst['dst_value'].mean())
                except Exception:
                    pass

        if events_src[i] == 1 and mags_src[i] >= MW_THRESHOLD:
            events_new[i] = 1
            mags_new[i]   = mags_src[i]
            n_kept += 1
            eq_info = eq_catalog.get(dk)
            if eq_info:
                eq_lat, eq_lon = eq_info['lat'], eq_info['lon']
                for station, (slat, slon) in station_coords.items():
                    if station in sta_idx:
                        si = sta_idx[station]
                        azms_new[i, si]  = azimuth_calc(slat, slon, eq_lat, eq_lon)
                        dists_new[i, si] = haversine(slat, slon, eq_lat, eq_lon)
            else:
                # Remap from V11 azm (per-station scalar)
                for s in src_stations:
                    if s in sta_idx:
                        oi = src_stations.index(s)
                        ni = sta_idx[s]
                        if oi < azms_src.shape[1]:
                            azms_new[i, ni] = azms_src[i, oi]
        elif events_src[i] == 1:
            n_downgraded += 1

    n_ev = int(events_new.sum())
    n_no = n_days - n_ev
    print(f"    Events (Mw>={MW_THRESHOLD}): {n_ev} | Noise: {n_no} | Downgraded: {n_downgraded}")

    # Data balancing: undersample noise if ratio > TARGET_RATIO
    keep_indices = list(range(n_days))
    if n_ev > 0 and n_no > n_ev * TARGET_RATIO:
        ev_idx    = [i for i in range(n_days) if events_new[i] == 1]
        noise_idx = [i for i in range(n_days) if events_new[i] == 0]
        target_n  = n_ev * TARGET_RATIO
        noise_df  = pd.DataFrame({'idx': noise_idx,
                                  'date': [dates_str[i] for i in noise_idx]})
        noise_df['month'] = noise_df['date'].str[:7]
        n_months  = noise_df['month'].nunique()
        spm       = max(1, int(target_n / n_months))
        sampled   = (noise_df.groupby('month')
                     .apply(lambda x: x.sample(min(spm, len(x)), random_state=42))
                     .reset_index(drop=True)['idx'].tolist())
        keep_indices = sorted(ev_idx + sampled)
        n_no = len(sampled)
        print(f"    After balancing: {len(keep_indices)} days | {n_ev} events | {n_no} noise (1:{n_no/n_ev:.1f})")

    n_out = len(keep_indices)
    temp_file = f'temp_{split_name}_v13.h5'
    active_counts = []

    with h5py.File(temp_file, 'w') as tmp:
        tds = tmp.create_dataset('tensors',
            shape=(n_out, n_sta, 3, 128, 1440), dtype=np.float16,
            compression='gzip', chunks=(1, n_sta, 3, 128, 1440))

        for out_i, src_i in enumerate(keep_indices):
            if out_i % 50 == 0:
                print(f"    Writing day {out_i+1}/{n_out}  ({dates_str[src_i]})...")
            dk = dates_key[src_i]
            day = np.zeros((n_sta, 3, 128, 1440), dtype=np.float16)
            active = 0
            for station in all_stations:
                if station not in sta_idx:
                    continue
                ni = sta_idx[station]
                t = get_tensor_for_station(station, dk, src_lookup, rt_lookup, open_files)
                if t is not None:
                    day[ni] = t
                    active += 1
            tds[out_i] = day
            active_counts.append(active)

    avg_active = float(np.mean(active_counts)) if active_counts else 0.0
    print(f"    Avg active stations/day: {avg_active:.1f}/{n_sta}")

    return {
        'temp_file':   temp_file,
        'label_event': events_new[keep_indices],
        'label_mag':   mags_new[keep_indices],
        'label_azm':   azms_new[keep_indices],
        'label_dist':  dists_new[keep_indices],
        'cosmic':      cosmic_new[keep_indices],
        'dates':       np.array([dates_src[i] for i in keep_indices]),
        'stations':    all_stations,
        'n_days':      n_out,
        'avg_active':  avg_active,
    }


def tahap2_5_build_train_val(station_coords, all_stations):
    print("\n" + "=" * 70)
    print("TAHAP 2-5: BUILD TRAIN & VAL (24 stations, Mw >= 5.0)")
    print("=" * 70)

    eq_catalog = load_eq_catalog()
    print(f"  EQ catalog loaded: {len(eq_catalog)} event dates (Mw>={MW_THRESHOLD})")

    print("  Building source lookup tables...")
    src_lookup = build_source_lookup()
    rt_lookup  = build_raw_tensor_lookup()
    print(f"  Source HDF5 lookup: {len(src_lookup)} entries")
    print(f"  Raw tensor lookup : {len(rt_lookup)} entries")

    # Open source HDF5 files
    open_files = {}
    for fname in SOURCE_HDF5:
        if os.path.exists(fname):
            open_files[fname] = h5py.File(fname, 'r')

    results = {}
    for split_name in ['train', 'val']:
        print(f"\n  Processing {split_name.upper()}...")
        data = build_split(split_name, station_coords, all_stations,
                           src_lookup, rt_lookup, open_files, eq_catalog)
        results[split_name] = data

    # Close source files
    for f in open_files.values():
        f.close()

    # Save to V13 train/val file
    print(f"\n  Saving {V13_TRAIN_VAL}...")
    with h5py.File(V13_TRAIN_VAL, 'w') as out_f:
        for split_name, data in results.items():
            grp = out_f.create_group(split_name)
            with h5py.File(data['temp_file'], 'r') as tmp:
                tmp.copy('tensors', grp)
            grp.create_dataset('label_event',    data=data['label_event'])
            grp.create_dataset('label_mag',      data=data['label_mag'])
            grp.create_dataset('label_azm',      data=data['label_azm'])
            grp.create_dataset('label_dist',     data=data['label_dist'])
            grp.create_dataset('cosmic_features',data=data['cosmic'])
            grp.create_dataset('dates',          data=data['dates'])
            grp.attrs['num_stations']        = len(data['stations'])
            grp.attrs['stations']            = data['stations']
            grp.attrs['mw_threshold']        = MW_THRESHOLD
            grp.attrs['avg_active_stations'] = data['avg_active']
            os.remove(data['temp_file'])

            n_ev = int(data['label_event'].sum())
            n_no = data['n_days'] - n_ev
            ratio = n_no / n_ev if n_ev > 0 else float('inf')
            print(f"  [{split_name.upper()}] {data['n_days']} days | {n_ev} events ({n_ev/data['n_days']*100:.0f}%) | {n_no} noise | ratio 1:{ratio:.1f} | avg_active {data['avg_active']:.1f}")

        out_f.attrs['created']      = datetime.now().isoformat()
        out_f.attrs['version']      = 'v13'
        out_f.attrs['mw_threshold'] = MW_THRESHOLD
        out_f.attrs['train_end']    = str(TRAIN_END)
        out_f.attrs['val_start']    = str(VAL_START)
        out_f.attrs['val_end']      = str(VAL_END)

    size_mb = os.path.getsize(V13_TRAIN_VAL) / (1024**2)
    print(f"  Saved: {V13_TRAIN_VAL}  ({size_mb:.1f} MB)")


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
                n_ev  = int(events.sum())
                n_no  = len(events) - n_ev
                ratio = n_no / n_ev if n_ev > 0 else float('inf')
                ev_mags = mags[events == 1]
                mag_str = f"{ev_mags.min():.2f}-{ev_mags.max():.2f}" if len(ev_mags) > 0 else "N/A"
                avg_act = g.attrs.get('avg_active_stations', 'N/A')
                cosmic  = g['cosmic_features'][:]
                # NaN check
                sample  = g['tensors'][:3]
                nan_cnt = int(np.isnan(sample).sum())
                # Active station check
                t0 = g['tensors'][0]
                active_in_day0 = int(np.any(t0.reshape(t0.shape[0], -1) != 0, axis=1).sum())

                print(f"    [{grp_name.upper()}]")
                print(f"      Days        : {len(dates)}")
                print(f"      Events      : {n_ev} ({n_ev/len(dates)*100:.0f}%)")
                print(f"      Noise       : {n_no} ({n_no/len(dates)*100:.0f}%)")
                print(f"      Ratio       : 1:{ratio:.1f}")
                print(f"      Mag range   : {mag_str} Mw")
                print(f"      Stations    : {len(stations)}")
                print(f"      Avg active  : {avg_act}")
                print(f"      Active day0 : {active_in_day0}/{len(stations)} stations")
                print(f"      Date range  : {min(dates)} to {max(dates)}")
                print(f"      Kp range    : {cosmic[:,0].min():.2f} to {cosmic[:,0].max():.2f}")
                print(f"      Dst range   : {cosmic[:,1].min():.2f} to {cosmic[:,1].max():.2f}")
                print(f"      NaN check   : {nan_cnt} NaN in first 3 days")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("V13 FULL PIPELINE (CORRECTED)")
    print(f"  Mw threshold : >= {MW_THRESHOLD}")
    print(f"  Train period : <= {TRAIN_END}")
    print(f"  Val period   : {VAL_START} to {VAL_END}")
    print(f"  Target ratio : 1:{TARGET_RATIO}")
    print("=" * 70)

    station_coords = load_stations()
    all_stations   = sorted(station_coords.keys())
    print(f"\nStations loaded: {len(station_coords)}")
    print(f"  {all_stations}")

    # TAHAP 1
    tahap1_calibrate_blindtest(station_coords, all_stations)

    # TAHAP 2-5
    tahap2_5_build_train_val(station_coords, all_stations)

    # Report
    final_report()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  {V13_BLINDTEST}")
    print(f"  {V13_TRAIN_VAL}")


if __name__ == '__main__':
    main()
