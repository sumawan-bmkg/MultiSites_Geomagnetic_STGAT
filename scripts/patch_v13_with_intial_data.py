"""
Patch V13 Datasets using Support Files from intial/
====================================================
Fixes:
  1. Dst values: replace -15.0 (suspicious) with real values from intial/dst.txt
  2. Azimuth/Distance: compute from intial/earthquake_catalog_2018_2025_merged.csv
  3. EQ catalog 2026: from intial/EQ1-4.2026.csv

Outputs:
  dataset_v13_train_val_M5_patched.h5
  dataset_v13_blindtest_M5_patched.h5
"""

import os
import h5py
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2, degrees


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
    """Load 24 station coordinates from lokasi_stasiun.csv."""
    stations = {}
    with open('intial/lokasi_stasiun.csv', 'r', encoding='utf-8-sig') as f:
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
    """
    Parse intial/dst.txt -> daily mean Dst per date.
    Returns dict: {'YYYY-MM-DD': dst_mean_float}
    """
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
    daily = dst_df.groupby('date_str')['dst'].mean().to_dict()
    return daily


def load_eq_catalog():
    """
    Load all EQ catalogs (2018-2026) with Mw >= 5.0.
    Returns dict: {'YYYY-MM-DD': {'mag': float, 'lat': float, 'lon': float}}
    """
    eq_by_date = {}

    # Historical 2018-2025
    for fname in [
        'intial/earthquake_catalog_2018_2025_merged.csv',
        'intial/EQ Repository  september Requested Data.csv',
        'intial/EQ Repository oktober Requested Data.csv',
        'intial/EQ Repository november  Requested Data.csv',
        'intial/EQ Repository desember Requested Data.csv',
    ]:
        if not os.path.exists(fname):
            continue
        df = pd.read_csv(fname)
        for col in ['datetime', 'Date time']:
            if col in df.columns:
                df['dt'] = pd.to_datetime(df[col], utc=True, errors='coerce')
                break
        if 'dt' not in df.columns:
            continue
        df = df.dropna(subset=['dt', 'Latitude', 'Longitude', 'Magnitude'])
        df = df[df['Magnitude'] >= 5.0]
        for _, row in df.iterrows():
            ds = row['dt'].strftime('%Y-%m-%d')
            if ds not in eq_by_date or row['Magnitude'] > eq_by_date[ds]['mag']:
                eq_by_date[ds] = {
                    'mag': float(row['Magnitude']),
                    'lat': float(row['Latitude']),
                    'lon': float(row['Longitude'])
                }

    # 2026 catalogs
    for fname in ['intial/EQ1.2026.csv', 'intial/EQ2.2026.csv',
                  'intial/EQ3.2026.csv', 'intial/EQ4.2026.csv']:
        if not os.path.exists(fname):
            continue
        df = pd.read_csv(fname)
        df['dt'] = pd.to_datetime(df['Date time'], utc=True, errors='coerce')
        df = df.dropna(subset=['dt', 'Latitude', 'Longitude', 'Magnitude'])
        df = df[df['Magnitude'] >= 5.0]
        for _, row in df.iterrows():
            ds = row['dt'].strftime('%Y-%m-%d')
            if ds not in eq_by_date or row['Magnitude'] > eq_by_date[ds]['mag']:
                eq_by_date[ds] = {
                    'mag': float(row['Magnitude']),
                    'lat': float(row['Latitude']),
                    'lon': float(row['Longitude'])
                }

    return eq_by_date


# ============================================================================
# PATCH FUNCTION
# ============================================================================

def patch_dataset(src_fname, out_fname, station_coords, all_stations,
                  dst_lookup, eq_catalog):
    """Patch a V13 dataset file with real Dst and azm/dist from catalogs."""
    print(f"\n  Patching: {src_fname}")
    n_sta   = len(all_stations)
    sta_idx = {s: i for i, s in enumerate(all_stations)}

    with h5py.File(src_fname, 'r') as src, h5py.File(out_fname, 'w') as dst_f:
        # Copy global attributes
        for k, v in src.attrs.items():
            dst_f.attrs[k] = v
        dst_f.attrs['patched']      = 'true'
        dst_f.attrs['patch_source'] = 'intial/dst.txt + earthquake_catalog_2018_2025_merged.csv'

        for grp_name in src.keys():
            g_src = src[grp_name]
            g_dst = dst_f.create_group(grp_name)

            # Copy tensors (largest dataset) directly
            print(f"    [{grp_name}] copying tensors...")
            src.copy(f'{grp_name}/tensors', g_dst)

            # Load labels
            events    = g_src['label_event'][:]
            mags      = g_src['label_mag'][:]
            azms      = g_src['label_azm'][:].copy()
            dists     = (g_src['label_dist'][:].copy()
                         if 'label_dist' in g_src
                         else np.zeros_like(azms))
            cosmic    = g_src['cosmic_features'][:].copy()
            dates_raw = g_src['dates'][:]
            dates     = [d.decode() if isinstance(d, bytes) else d
                         for d in dates_raw]

            n_days = len(dates)
            n_dst_fixed = 0
            n_azm_fixed = 0
            n_azm_missing = 0

            for i, date_str in enumerate(dates):
                # --- Fix Dst ---
                if date_str in dst_lookup:
                    old_dst = float(cosmic[i, 1])
                    if old_dst == -15.0 or old_dst == 0.0:
                        cosmic[i, 1] = float(dst_lookup[date_str])
                        n_dst_fixed += 1

                # --- Fix azm/dist for event days ---
                if events[i] == 1:
                    if date_str in eq_catalog:
                        eq = eq_catalog[date_str]
                        eq_lat, eq_lon = eq['lat'], eq['lon']
                        new_azm  = np.zeros(n_sta, dtype=np.float32)
                        new_dist = np.zeros(n_sta, dtype=np.float32)
                        for station, (slat, slon) in station_coords.items():
                            if station in sta_idx:
                                si = sta_idx[station]
                                new_azm[si]  = azimuth_calc(slat, slon, eq_lat, eq_lon)
                                new_dist[si] = haversine(slat, slon, eq_lat, eq_lon)
                        azms[i]  = new_azm
                        dists[i] = new_dist
                        n_azm_fixed += 1
                    else:
                        n_azm_missing += 1

            print(f"    [{grp_name}] Dst fixed: {n_dst_fixed}/{n_days} | "
                  f"azm/dist fixed: {n_azm_fixed}/{int(events.sum())} | "
                  f"azm missing: {n_azm_missing}")

            # Save patched labels
            g_dst.create_dataset('label_event',    data=events)
            g_dst.create_dataset('label_mag',      data=mags)
            g_dst.create_dataset('label_azm',      data=azms)
            g_dst.create_dataset('label_dist',     data=dists)
            g_dst.create_dataset('cosmic_features',data=cosmic)
            g_dst.create_dataset('dates',          data=dates_raw)

            # Copy group attributes
            for k, v in g_src.attrs.items():
                g_dst.attrs[k] = v

    size_mb = os.path.getsize(out_fname) / (1024**2)
    print(f"    Saved: {out_fname}  ({size_mb:.1f} MB)")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_patched(fname):
    """Print validation summary for a patched dataset."""
    print(f"\n  Validating: {fname}")
    with h5py.File(fname, 'r') as f:
        for grp_name in f.keys():
            g = f[grp_name]
            events  = g['label_event'][:]
            mags    = g['label_mag'][:]
            azms    = g['label_azm'][:]
            dists   = g['label_dist'][:]
            cosmic  = g['cosmic_features'][:]
            dates_r = g['dates'][:]
            dates   = [d.decode() if isinstance(d, bytes) else d for d in dates_r]

            n_ev = int(events.sum())
            n_no = len(events) - n_ev

            # Dst quality
            dst_suspicious = int((cosmic[:, 1] == -15.0).sum())
            dst_zero       = int((cosmic[:, 1] == 0.0).sum())
            dst_valid      = len(dates) - dst_suspicious - dst_zero

            # Azm quality (event days)
            ev_azms = azms[events == 1]
            azm_nonzero = int(np.any(ev_azms != 0, axis=1).sum()) if len(ev_azms) > 0 else 0

            # Dist quality
            ev_dists = dists[events == 1]
            dist_nonzero = int(np.any(ev_dists != 0, axis=1).sum()) if len(ev_dists) > 0 else 0

            print(f"    [{grp_name.upper()}]")
            print(f"      Days     : {len(dates)} | Events: {n_ev} | Noise: {n_no}")
            print(f"      Dst valid: {dst_valid}/{len(dates)} | suspicious(-15): {dst_suspicious} | zero: {dst_zero}")
            print(f"      Azm fixed: {azm_nonzero}/{n_ev} event days have non-zero azimuth")
            print(f"      Dist fixed: {dist_nonzero}/{n_ev} event days have non-zero distance")
            print(f"      Kp range : {cosmic[:,0].min():.2f} to {cosmic[:,0].max():.2f}")
            print(f"      Dst range: {cosmic[:,1].min():.2f} to {cosmic[:,1].max():.2f}")
            print(f"      NaN check: {int(np.isnan(g['tensors'][:3]).sum())} NaN in first 3 days")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PATCH V13 DATASETS WITH SUPPORT FILES FROM intial/")
    print("=" * 70)

    # Load support data
    print("\nLoading support data from intial/...")
    station_coords = load_stations()
    all_stations   = sorted(station_coords.keys())
    dst_lookup     = load_dst_lookup()
    eq_catalog     = load_eq_catalog()

    print(f"  Stations     : {len(station_coords)} -> {all_stations}")
    print(f"  Dst records  : {len(dst_lookup)} daily averages (2018-2024)")
    print(f"  EQ catalog   : {len(eq_catalog)} event dates (Mw>=5.0)")

    # Patch train/val
    patch_dataset(
        'dataset_v13_train_val_M5.h5',
        'dataset_v13_train_val_M5_patched.h5',
        station_coords, all_stations, dst_lookup, eq_catalog
    )

    # Patch blindtest
    patch_dataset(
        'dataset_v13_blindtest_M5.h5',
        'dataset_v13_blindtest_M5_patched.h5',
        station_coords, all_stations, dst_lookup, eq_catalog
    )

    # Validate
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    validate_patched('dataset_v13_train_val_M5_patched.h5')
    validate_patched('dataset_v13_blindtest_M5_patched.h5')

    print("\n" + "=" * 70)
    print("PATCH COMPLETE")
    print("=" * 70)
    for fname in ['dataset_v13_train_val_M5_patched.h5',
                  'dataset_v13_blindtest_M5_patched.h5']:
        if os.path.exists(fname):
            size_mb = os.path.getsize(fname) / (1024**2)
            print(f"  {fname}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
