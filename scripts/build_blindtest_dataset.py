"""
Build Blindtest Dataset from Blindtest Scalograms + EQ Catalog 2026
====================================================================
Tujuan:
  - Gabungkan semua scalogram dari blindtest/scalogram/ (2026-01-01 to 2026-05-01)
  - Cocokkan dengan katalog gempa 2026 dari intial/EQ1-4.2026.csv
  - Hitung label: event, magnitude, azimuth per stasiun, jarak Haversine
  - Integrasikan Kp dan Dst dari blindtest/kp_index/
  - Simpan sebagai dataset_v12_blindtest.h5 dengan format multi-station graph

Format output:
  dataset_v12_blindtest.h5
  └── test/
      ├── tensors:         (N_days, N_stations, 3, 128, 1440) float16
      ├── label_event:     (N_days,) int8
      ├── label_mag:       (N_days,) float32
      ├── label_azm:       (N_days, N_stations) float32
      ├── label_dist:      (N_days, N_stations) float32
      ├── cosmic_features: (N_days, 2) float32  [Kp, Dst]
      └── dates:           (N_days,) S10
"""

import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, date
from math import radians, sin, cos, sqrt, atan2, degrees
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCALOGRAM_DIR  = 'blindtest/scalogram'
KP_INDEX_DIR   = 'blindtest/kp_index'
EQ_CATALOGS    = [
    'intial/EQ1.2026.csv',
    'intial/EQ2.2026.csv',
    'intial/EQ3.2026.csv',
    'intial/EQ4.2026.csv',
]
STATION_FILE   = 'intial/lokasi_stasiun.csv'
OUTPUT_FILE    = 'dataset_v12_blindtest.h5'

MIN_MAGNITUDE  = 4.0
BLINDTEST_START = date(2026, 1, 1)


# ============================================================================
# HELPER: Haversine distance (km)
# ============================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


# ============================================================================
# HELPER: Azimuth from station to epicenter (degrees)
# ============================================================================

def azimuth(lat1, lon1, lat2, lon2):
    phi1, phi2 = radians(lat1), radians(lat2)
    dlam = radians(lon2 - lon1)
    x = sin(dlam) * cos(phi2)
    y = cos(phi1)*sin(phi2) - sin(phi1)*cos(phi2)*cos(dlam)
    az = degrees(atan2(x, y))
    return (az + 360) % 360


# ============================================================================
# STEP 1: Load station coordinates
# ============================================================================

def load_stations():
    print("=" * 70)
    print("STEP 1: LOAD STATION COORDINATES")
    print("=" * 70)

    # Parse the semicolon-separated file (handle non-breaking spaces and BOM)
    stations = {}
    with open(STATION_FILE, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip header
        # Remove non-breaking spaces and other whitespace variants
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

    print(f"  Loaded {len(stations)} stations:")
    for code, (lat, lon) in sorted(stations.items()):
        print(f"    {code}: ({lat:.4f}, {lon:.4f})")

    return stations


# ============================================================================
# STEP 2: Load and merge EQ catalogs 2026
# ============================================================================

def load_eq_catalog():
    print("\n" + "=" * 70)
    print("STEP 2: LOAD EARTHQUAKE CATALOG 2026")
    print("=" * 70)

    dfs = []
    for fname in EQ_CATALOGS:
        if not os.path.exists(fname):
            print(f"  WARNING: {fname} not found")
            continue
        df = pd.read_csv(fname)
        dfs.append(df)
        print(f"  Loaded {fname}: {len(df)} events")

    all_eq = pd.concat(dfs, ignore_index=True)
    all_eq['datetime'] = pd.to_datetime(all_eq['Date time'], utc=True)
    all_eq['date'] = all_eq['datetime'].dt.date

    # Filter Mw >= 4.0
    all_eq = all_eq[all_eq['Magnitude'] >= MIN_MAGNITUDE].copy()
    all_eq = all_eq.sort_values('datetime').reset_index(drop=True)

    # Filter only blindtest period (>= 2026-01-01)
    all_eq = all_eq[all_eq['date'] >= BLINDTEST_START].copy()

    print(f"\n  Total events (Mw >= {MIN_MAGNITUDE}, >= 2026-01-01): {len(all_eq)}")
    print(f"  Date range: {all_eq['date'].min()} to {all_eq['date'].max()}")
    print(f"  Magnitude range: {all_eq['Magnitude'].min():.2f} to {all_eq['Magnitude'].max():.2f}")
    print(f"  Unique event dates: {all_eq['date'].nunique()}")

    # Group by date: take the largest magnitude event per day
    daily_eq = all_eq.groupby('date').apply(
        lambda g: g.loc[g['Magnitude'].idxmax()]
    ).reset_index(drop=True)

    print(f"\n  Daily events (max mag per day): {len(daily_eq)}")

    return daily_eq


# ============================================================================
# STEP 3: Load Kp and Dst for a given date
# ============================================================================

def load_cosmic_for_date(date_str):
    """
    Load Kp and Dst for a given date (YYYYMMDD).
    Returns (kp_max, dst_mean).
    """
    # Try DST file: dst_index_YYYYMMDD.csv
    dst_file = os.path.join(KP_INDEX_DIR, f'dst_index_{date_str}.csv')
    dst_val = 0.0
    if os.path.exists(dst_file):
        try:
            df = pd.read_csv(dst_file)
            if 'dst_value' in df.columns:
                dst_val = float(df['dst_value'].mean())
        except Exception:
            pass

    # Try KP file by timestamp: kp_index_TIMESTAMP.csv
    # Convert date to timestamp (midnight UTC)
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        ts = int(dt.timestamp())
        kp_file = os.path.join(KP_INDEX_DIR, f'kp_index_{ts}.csv')
        kp_val = 0.0
        if os.path.exists(kp_file):
            df = pd.read_csv(kp_file)
            if 'kp_value' in df.columns:
                kp_val = float(df['kp_value'].max())
        else:
            # Try to find any kp file for this date by scanning
            kp_val = 0.0
            for fname in os.listdir(KP_INDEX_DIR):
                if fname.startswith('kp_index_') and fname.endswith('.csv'):
                    try:
                        file_ts = int(fname.replace('kp_index_', '').replace('.csv', ''))
                        file_dt = datetime.utcfromtimestamp(file_ts).date()
                        if str(file_dt).replace('-', '') == date_str:
                            df = pd.read_csv(os.path.join(KP_INDEX_DIR, fname))
                            if 'kp_value' in df.columns:
                                kp_val = max(kp_val, float(df['kp_value'].max()))
                    except Exception:
                        pass
    except Exception:
        kp_val = 0.0

    return kp_val, dst_val


# ============================================================================
# STEP 4: Inventory scalogram files
# ============================================================================

def inventory_scalograms():
    print("\n" + "=" * 70)
    print("STEP 3: INVENTORY SCALOGRAM FILES")
    print("=" * 70)

    files = [f for f in os.listdir(SCALOGRAM_DIR) if f.endswith('.h5')]
    records = {}  # {(date_str, station): filepath}

    for fname in files:
        parts = fname.replace('scalogram_', '').replace('.h5', '').split('_')
        if len(parts) == 2:
            station, date_str = parts
            records[(date_str, station)] = os.path.join(SCALOGRAM_DIR, fname)

    # Get unique dates and stations
    dates = sorted(set(k[0] for k in records.keys()))
    stations = sorted(set(k[1] for k in records.keys()))

    print(f"  Total scalogram files: {len(files)}")
    print(f"  Unique dates: {len(dates)} ({min(dates)} to {max(dates)})")
    print(f"  Stations ({len(stations)}): {stations}")

    # Check one file to get tensor shape
    sample_key = list(records.keys())[0]
    with h5py.File(records[sample_key], 'r') as f:
        station_code = sample_key[1]
        tensor = f[f'daily/{station_code}/tensors'][0]
        print(f"  Tensor shape per station: {tensor.shape}")

    return records, dates, stations


# ============================================================================
# STEP 5: Build multi-station graph snapshots
# ============================================================================

def build_blindtest_dataset(records, dates, stations, station_coords, daily_eq):
    print("\n" + "=" * 70)
    print("STEP 4: BUILD BLINDTEST DATASET")
    print("=" * 70)

    n_stations = len(stations)
    n_days = len(dates)
    sta_idx = {s: i for i, s in enumerate(stations)}

    # Build EQ lookup by date string
    eq_by_date = {}
    for _, row in daily_eq.iterrows():
        date_str = str(row['date']).replace('-', '')
        eq_by_date[date_str] = row

    print(f"  Days to process: {n_days}")
    print(f"  Stations: {n_stations}")
    print(f"  Event days in catalog: {len(eq_by_date)}")

    # Statistics
    n_events = 0
    n_noise = 0
    missing_scalograms = []

    with h5py.File(OUTPUT_FILE, 'w') as out_f:
        test_grp = out_f.create_group('test')

        # Create datasets
        tensors_ds = test_grp.create_dataset(
            'tensors',
            shape=(n_days, n_stations, 3, 128, 1440),
            dtype=np.float16,
            compression='gzip',
            chunks=(1, n_stations, 3, 128, 1440)
        )

        label_event  = np.zeros(n_days, dtype=np.int8)
        label_mag    = np.zeros(n_days, dtype=np.float32)
        label_azm    = np.zeros((n_days, n_stations), dtype=np.float32)
        label_dist   = np.zeros((n_days, n_stations), dtype=np.float32)
        cosmic_feat  = np.zeros((n_days, 2), dtype=np.float32)
        dates_out    = []

        for day_idx, date_str in enumerate(dates):
            if day_idx % 20 == 0:
                print(f"  Processing day {day_idx+1}/{n_days}: {date_str}...")

            # Build day tensor
            day_tensor = np.zeros((n_stations, 3, 128, 1440), dtype=np.float16)

            for station in stations:
                key = (date_str, station)
                if key not in records:
                    missing_scalograms.append(key)
                    continue
                try:
                    with h5py.File(records[key], 'r') as f:
                        grp_path = f'daily/{station}'
                        if grp_path in f:
                            tensor = f[f'{grp_path}/tensors'][0]  # (3, 128, 1440)
                            day_tensor[sta_idx[station]] = tensor.astype(np.float16)
                except Exception as e:
                    pass

            tensors_ds[day_idx] = day_tensor

            # Labels
            if date_str in eq_by_date:
                eq = eq_by_date[date_str]
                eq_lat = float(eq['Latitude'])
                eq_lon = float(eq['Longitude'])
                eq_mag = float(eq['Magnitude'])

                label_event[day_idx] = 1
                label_mag[day_idx]   = eq_mag
                n_events += 1

                # Azimuth and distance per station
                for station in stations:
                    if station in station_coords:
                        sta_lat, sta_lon = station_coords[station]
                        s_idx = sta_idx[station]
                        label_azm[day_idx, s_idx]  = azimuth(sta_lat, sta_lon, eq_lat, eq_lon)
                        label_dist[day_idx, s_idx] = haversine(sta_lat, sta_lon, eq_lat, eq_lon)
            else:
                n_noise += 1

            # Cosmic features (Kp, Dst)
            kp_val, dst_val = load_cosmic_for_date(date_str)
            cosmic_feat[day_idx, 0] = kp_val
            cosmic_feat[day_idx, 1] = dst_val

            # Date string
            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            dates_out.append(date_formatted)

        # Save labels
        test_grp.create_dataset('label_event',    data=label_event)
        test_grp.create_dataset('label_mag',      data=label_mag)
        test_grp.create_dataset('label_azm',      data=label_azm)
        test_grp.create_dataset('label_dist',     data=label_dist)
        test_grp.create_dataset('cosmic_features',data=cosmic_feat)
        test_grp.create_dataset('dates',          data=np.array(dates_out, dtype='S10'))

        # Attributes
        test_grp.attrs['num_stations'] = n_stations
        test_grp.attrs['stations']     = stations
        test_grp.attrs['n_events']     = n_events
        test_grp.attrs['n_noise']      = n_noise

        # Global attributes
        out_f.attrs['created']         = datetime.now().isoformat()
        out_f.attrs['version']         = 'v12'
        out_f.attrs['format']          = 'multi-station-graph-blindtest'
        out_f.attrs['blindtest_start'] = '2026-01-01'
        out_f.attrs['min_magnitude']   = MIN_MAGNITUDE
        out_f.attrs['source_scalogram']= SCALOGRAM_DIR
        out_f.attrs['source_catalog']  = 'intial/EQ1-4.2026.csv'

    print(f"\n  ✓ Dataset built:")
    print(f"    Total days  : {n_days}")
    print(f"    Event days  : {n_events} ({n_events/n_days*100:.1f}%)")
    print(f"    Noise days  : {n_noise} ({n_noise/n_days*100:.1f}%)")
    print(f"    Missing scalograms: {len(missing_scalograms)}")

    return n_events, n_noise


# ============================================================================
# STEP 6: Final validation and report
# ============================================================================

def validate_and_report():
    print("\n" + "=" * 70)
    print("STEP 5: VALIDATION & FINAL REPORT")
    print("=" * 70)

    with h5py.File(OUTPUT_FILE, 'r') as f:
        g = f['test']
        tensors   = g['tensors']
        events    = g['label_event'][:]
        mags      = g['label_mag'][:]
        azms      = g['label_azm'][:]
        dists     = g['label_dist'][:]
        cosmic    = g['cosmic_features'][:]
        dates_raw = g['dates'][:]
        stations  = list(g.attrs.get('stations', []))
        if stations and isinstance(stations[0], bytes):
            stations = [s.decode() for s in stations]
        dates = [d.decode() if isinstance(d, bytes) else d for d in dates_raw]

        n_ev = int(events.sum())
        n_no = len(events) - n_ev

        print(f"\n  BLINDTEST SET (dataset_v12_blindtest.h5):")
        print(f"    Shape     : {tensors.shape}")
        print(f"    Days      : {len(dates)}")
        print(f"    Date range: {min(dates)} to {max(dates)}")
        print(f"    Events    : {n_ev} ({n_ev/len(events)*100:.1f}%)")
        print(f"    Noise     : {n_no} ({n_no/len(events)*100:.1f}%)")
        print(f"    Ratio     : 1:{n_no/n_ev:.1f}" if n_ev > 0 else "    Ratio: N/A")
        print(f"    Stations  : {len(stations)} -> {stations}")

        # Magnitude stats for events
        ev_mags = mags[events == 1]
        if len(ev_mags) > 0:
            print(f"\n    Magnitude stats (events only):")
            print(f"      Min: {ev_mags.min():.2f} Mw")
            print(f"      Max: {ev_mags.max():.2f} Mw")
            print(f"      Mean: {ev_mags.mean():.2f} Mw")

        # Azimuth stats
        ev_azms = azms[events == 1]
        if len(ev_azms) > 0:
            valid_azms = ev_azms[ev_azms > 0]
            print(f"\n    Azimuth stats (event days, non-zero):")
            print(f"      Range: {valid_azms.min():.1f}° to {valid_azms.max():.1f}°")

        # Distance stats
        ev_dists = dists[events == 1]
        if len(ev_dists) > 0:
            valid_dists = ev_dists[ev_dists > 0]
            print(f"\n    Distance stats (event days, non-zero):")
            print(f"      Range: {valid_dists.min():.1f} km to {valid_dists.max():.1f} km")

        # Cosmic features
        print(f"\n    Cosmic features:")
        print(f"      Kp range: {cosmic[:, 0].min():.2f} to {cosmic[:, 0].max():.2f}")
        print(f"      Dst range: {cosmic[:, 1].min():.2f} to {cosmic[:, 1].max():.2f}")

        # NaN check
        sample = tensors[:5]
        print(f"\n    Data quality:")
        print(f"      NaN in tensors (first 5 days): {np.isnan(sample).sum()}")
        print(f"      NaN in events: {np.isnan(events.astype(float)).sum()}")
        print(f"      NaN in mags: {np.isnan(mags).sum()}")
        print(f"      NaN in azms: {np.isnan(azms).sum()}")
        print(f"      NaN in dists: {np.isnan(dists).sum()}")
        print(f"      NaN in cosmic: {np.isnan(cosmic).sum()}")

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"\n  Output file: {OUTPUT_FILE}")
    print(f"  File size  : {size_mb:.2f} MB")


# ============================================================================
# STEP 7: Create visualization
# ============================================================================

def create_visualization():
    print("\n" + "=" * 70)
    print("STEP 6: CREATE VISUALIZATION")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    with h5py.File(OUTPUT_FILE, 'r') as f:
        g = f['test']
        events    = g['label_event'][:]
        mags      = g['label_mag'][:]
        azms      = g['label_azm'][:]
        dists     = g['label_dist'][:]
        cosmic    = g['cosmic_features'][:]
        dates_raw = g['dates'][:]
        dates = [d.decode() if isinstance(d, bytes) else d for d in dates_raw]
        stations = list(g.attrs.get('stations', []))
        if stations and isinstance(stations[0], bytes):
            stations = [s.decode() for s in stations]

        n_ev = int(events.sum())
        n_no = len(events) - n_ev

        # Plot 1: Event/Noise distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(['Events', 'Noise'], [n_ev, n_no],
                color=['#e74c3c', '#3498db'], alpha=0.85, edgecolor='white')
        ax1.set_title(f'Blindtest Distribution\n({n_ev} events, {n_no} noise)', fontweight='bold')
        ax1.set_ylabel('Days')
        for i, v in enumerate([n_ev, n_no]):
            ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Magnitude distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ev_mags = mags[events == 1]
        if len(ev_mags) > 0:
            ax2.hist(ev_mags, bins=20, color='#e74c3c', alpha=0.8, edgecolor='white')
            ax2.axvline(ev_mags.mean(), color='black', linestyle='--', label=f'Mean={ev_mags.mean():.2f}')
            ax2.set_xlabel('Magnitude (Mw)')
            ax2.set_ylabel('Count')
            ax2.set_title('Magnitude Distribution\n(Event Days)', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Temporal distribution
        ax3 = fig.add_subplot(gs[0, 2])
        event_dates = [dates[i] for i, e in enumerate(events) if e == 1]
        noise_dates = [dates[i] for i, e in enumerate(events) if e == 0]
        ax3.scatter(range(len(event_dates)), [1]*len(event_dates),
                    c='#e74c3c', label=f'Events ({len(event_dates)})', alpha=0.7, s=20)
        ax3.scatter(range(len(noise_dates)), [0]*len(noise_dates),
                    c='#3498db', label=f'Noise ({len(noise_dates)})', alpha=0.5, s=10)
        ax3.set_title('Temporal Distribution', fontweight='bold')
        ax3.set_xlabel('Day Index')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Azimuth heatmap (event days)
        ax4 = fig.add_subplot(gs[1, 0])
        ev_idx = np.where(events == 1)[0][:20]  # first 20 event days
        if len(ev_idx) > 0:
            azm_sample = azms[ev_idx]
            im = ax4.imshow(azm_sample.T, aspect='auto', cmap='twilight',
                           vmin=0, vmax=360)
            ax4.set_xlabel('Event Day Index')
            ax4.set_ylabel('Station Index')
            ax4.set_title('Azimuth per Station\n(First 20 Event Days)', fontweight='bold')
            plt.colorbar(im, ax=ax4, label='Azimuth (°)')

        # Plot 5: Distance heatmap (event days)
        ax5 = fig.add_subplot(gs[1, 1])
        if len(ev_idx) > 0:
            dist_sample = dists[ev_idx]
            im = ax5.imshow(dist_sample.T, aspect='auto', cmap='YlOrRd')
            ax5.set_xlabel('Event Day Index')
            ax5.set_ylabel('Station Index')
            ax5.set_title('Distance per Station (km)\n(First 20 Event Days)', fontweight='bold')
            plt.colorbar(im, ax=ax5, label='Distance (km)')

        # Plot 6: Cosmic features
        ax6 = fig.add_subplot(gs[1, 2])
        ev_cosmic = cosmic[events == 1]
        no_cosmic = cosmic[events == 0]
        ax6.scatter(no_cosmic[:, 0], no_cosmic[:, 1], alpha=0.5, s=15,
                    c='#3498db', label='Noise')
        ax6.scatter(ev_cosmic[:, 0], ev_cosmic[:, 1], alpha=0.7, s=25,
                    c='#e74c3c', label='Events')
        ax6.set_xlabel('Kp Index')
        ax6.set_ylabel('Dst Index')
        ax6.set_title('Space Weather Features', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.suptitle('Dataset V12 Blindtest (2026) - Validation Report',
                 fontsize=14, fontweight='bold')
    out_png = 'plots/v12_blindtest_report.png'
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_png}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("BUILD BLINDTEST DATASET V12 (2026)")
    print("=" * 70)
    print(f"  Scalogram dir : {SCALOGRAM_DIR}")
    print(f"  EQ catalogs   : {EQ_CATALOGS}")
    print(f"  Output        : {OUTPUT_FILE}")
    print(f"  Min magnitude : {MIN_MAGNITUDE}")

    # Step 1: Load stations
    station_coords = load_stations()

    # Step 2: Load EQ catalog
    daily_eq = load_eq_catalog()

    # Step 3: Inventory scalograms
    records, dates, stations = inventory_scalograms()

    # Step 4: Build dataset
    n_events, n_noise = build_blindtest_dataset(
        records, dates, stations, station_coords, daily_eq
    )

    # Step 5: Validate
    validate_and_report()

    # Step 6: Visualize
    create_visualization()

    print("\n" + "=" * 70)
    print("✅ BLINDTEST DATASET COMPLETE")
    print("=" * 70)
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Events: {n_events} days")
    print(f"  Noise : {n_noise} days")
    print(f"  Total : {n_events + n_noise} days")


if __name__ == '__main__':
    main()
