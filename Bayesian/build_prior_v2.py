#!/usr/bin/env python3
"""
build_prior_v2.py  --  MODULE 1: STATION-SPECIFIC DYNAMIC PRIORS
================================================================
Tujuan:
  Membangun prior azimuth spesifik untuk SETIAP stasiun geomagnetik.
  Menghitung bearing (azimuth) dari koordinat stasiun ke seluruh gempa 
  historis di 2026.csv, lalu generate KDE PDF 1D (360-D array) per stasiun.

Output:
  - prior_[station_id].pt    : torch.Tensor [360] (prior per stasiun)
  - prior_distribution_polar_[station_id].png : visualisasi per stasiun
"""

import argparse
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------------------------
# Konstanta
# ------------------------------------------------------------------------------
BINS = 360          # resolusi sudut 1 deg/bin
DEFAULT_BW = 15.0   # bandwidth Gaussian KDE dalam derajat

# Database koordinat stasiun geomagnetik Indonesia (BMKG)
# Format: station_id: (latitude, longitude)
STATION_COORDINATES = {
    'ALR': (-8.14, 124.59),  # Alor Island
    'AMB': (-3.68, 128.18),  # Ambon
    'CLP': (-1.01, 103.62),  # Cilacap
    'GTO': (-0.20, 100.31),  # Gunung Tabur
    'KPY': (-3.68, 102.58),  # Kepahiang
    'LPS': (-6.79, 111.49),  # Los Palos
    'LUT': (-2.54, 140.65),  # Lautem
    'LWA': (-6.32, 106.75),  # Liwa
    'LWK': (-1.95, 130.25),  # Lewoleba
    'MLB': (-1.48, 127.48),  # Labuha
    'PLU': (-6.89, 110.42),  # Pelabuhan Ratu
    'ROT': (-8.13, 123.55),  # Ruteng
    'SBG': (-4.14, 114.33),  # Sambas
    'SCN': (-6.12, 106.58),  # Santa Cruz
    'SKB': (-1.24, 116.87),  # Sambali Kebun
    'SMI': (-6.31, 106.74),  # Sukabumi
    'SRG': (-6.80, 111.52),  # Surabaya (Gresik)
    'SRO': (-0.86, 131.26),  # Sorong
    'TNT': (-6.17, 106.63),  # Tangerang
    'TRD': (-6.89, 107.61),  # Bandung (Lembang)
    'TRT': (-2.95, 104.65),  # Tanjung Karang
    'YOG': (-7.78, 110.36),  # Yogyakarta
}

# ------------------------------------------------------------------------------
# Helfer: geodesic bearing (azimuth) dua titik di permukaan bumi
# ------------------------------------------------------------------------------
def geodesic_azimuth(lat1: float, lon1: float,
                     lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dl   = math.radians(lon2 - lon1)

    y = math.sin(dl) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dl))

    az = math.atan2(y, x)
    return (math.degrees(az) + 360.0) % 360.0


# ------------------------------------------------------------------------------
# Gaussian KDE 1D untuk data siklik (circular)
# ------------------------------------------------------------------------------
def circular_kde_1d(azimuths_deg: np.ndarray,
                    bandwidth_deg: float = DEFAULT_BW,
                    bins: int = BINS) -> np.ndarray:
    centers = np.linspace(0.0, 360.0, bins, endpoint=False)
    sigma   = bandwidth_deg
    pdf     = np.zeros(bins, dtype=np.float64)

    for az in azimuths_deg:
        diff = centers - az
        diff = (diff + 180.0) % 360.0 - 180.0
        pdf += np.exp(-0.5 * (diff / sigma) ** 2)

    total = pdf.sum()
    if total < 1e-12:
        pdf = np.ones(bins, dtype=np.float64) / bins
    else:
        pdf /= total

    return pdf.astype(np.float32)


# ------------------------------------------------------------------------------
# Visualisasi polar per stasiun
# ------------------------------------------------------------------------------
def plot_polar_prior_station(pdf: np.ndarray,
                             station_id: str,
                             stn_lat: float,
                             stn_lon: float,
                             output_path: Path,
                             raw_azimuths: np.ndarray = None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[WARN] matplotlib tidak tersedia -- melewati visualisasi untuk {station_id}.")
        return

    bins    = len(pdf)
    angles  = np.linspace(0.0, 2 * np.pi, bins, endpoint=False)

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw=dict(projection='polar')
    )
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    ax.fill(angles, pdf, alpha=0.35, color='royalblue', label=f'KDE Prior {station_id}')
    ax.plot(angles, pdf, color='royalblue', linewidth=2.0)

    if raw_azimuths is not None and len(raw_azimuths) > 0:
        az_rad = np.radians(raw_azimuths)
        max_r  = pdf.max()
        r_vals = np.full_like(az_rad, max_r * 0.08)
        ax.scatter(az_rad, r_vals, color='tomato', s=10, alpha=0.5,
                   zorder=5, label=f'Raw events (n={len(raw_azimuths)})')

    ax.set_title(
        f"Station-Specific Spatial Prior -- {station_id} ({stn_lat:.4f}°N, {stn_lon:.4f}°E)",
        pad=18, fontsize=13, fontweight='bold'
    )
    ax.legend(loc='lower right', fontsize=10)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Polar plot tersimpan -> {output_path}")


# ------------------------------------------------------------------------------
# Fungsi utama: compute_station_kde
# ------------------------------------------------------------------------------
def compute_station_kde(station_coords: tuple, catalog_df: pd.DataFrame, 
                        bandwidth: float = DEFAULT_BW) -> np.ndarray:
    """
    Hitung KDE PDF 1D untuk satu stasiun berdasarkan koordinatnya.
    
    Args:
        station_coords: (lat, lon) koordinat stasiun
        catalog_df: DataFrame katalog gempa dengan kolom 'latitude', 'longitude'
        bandwidth: bandwidth KDE dalam derajat
    
    Returns:
        pdf: numpy array [360] probabilitas azimuth
    """
    stn_lat, stn_lon = station_coords
    
    # Filter valid events
    valid_df = catalog_df.dropna(subset=['latitude', 'longitude'])
    
    # Hitung azimuth dari stasiun ke setiap gempa
    azimuths = np.array([
        geodesic_azimuth(stn_lat, stn_lon, row['latitude'], row['longitude'])
        for _, row in valid_df.iterrows()
    ], dtype=np.float32)
    
    # Generate KDE
    pdf = circular_kde_1d(azimuths, bandwidth_deg=bandwidth, bins=BINS)
    
    return pdf, azimuths


# ------------------------------------------------------------------------------
# Main: Build station-specific priors
# ------------------------------------------------------------------------------
def build_station_priors(catalog_path: str,
                        bandwidth: float,
                        output_dir: str) -> None:

    cat_path = Path(catalog_path)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Membaca katalog: {cat_path}")
    try:
        df = pd.read_csv(cat_path, skiprows=2, quotechar='"', on_bad_lines='skip')
    except Exception as e:
        sys.exit(f"[ERROR] Gagal membaca CSV: {e}")

    df.columns = [c.strip().lower() for c in df.columns]
    lat_col = next((c for c in df.columns if 'lat' in c), None)
    lon_col = next((c for c in df.columns if 'lon' in c), None)

    if lat_col is None or lon_col is None:
        sys.exit(f"[ERROR] Kolom Latitude/Longitude tidak ditemukan.")

    df = df.dropna(subset=[lat_col, lon_col])
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col])

    print(f"[INFO] Total event valid: {len(df)}")
    if len(df) == 0:
        sys.exit("[ERROR] Tidak ada data event yang valid.")

    # Rename columns for consistency
    df = df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})

    print(f"[INFO] Membangun priors untuk {len(STATION_COORDINATES)} stasiun...")
    
    priors_built = 0
    
    for station_id, (stn_lat, stn_lon) in STATION_COORDINATES.items():
        print(f"\n[INFO] Processing station: {station_id} ({stn_lat:.4f}°N, {stn_lon:.4f}°E)")
        
        # Hitung KDE untuk stasiun ini
        pdf, raw_azimuths = compute_station_kde((stn_lat, stn_lon), df, bandwidth)
        
        print(f"[INFO] Azimuth stats -- mean={np.mean(raw_azimuths):.1f}deg  std={np.std(raw_azimuths):.1f}deg")
        print(f"[INFO] PDF valid -- sum={pdf.sum():.6f}")

        # Simpan tensor
        prior_tensor = torch.tensor(pdf, dtype=torch.float32)
        tensor_path  = out_dir / f"prior_{station_id}.pt"
        torch.save(prior_tensor, str(tensor_path))
        print(f"[OK] Tensor tersimpan -> {tensor_path}  (shape={prior_tensor.shape})")

        # Simpan metadata per stasiun
        meta_path = out_dir / f"prior_{station_id}_metadata.txt"
        with open(meta_path, "w") as f:
            f.write(f"station_id    : {station_id}\n")
            f.write(f"station_lat   : {stn_lat}\n")
            f.write(f"station_lon   : {stn_lon}\n")
            f.write(f"catalog       : {cat_path.resolve()}\n")
            f.write(f"bandwidth_deg : {bandwidth}\n")
            f.write(f"n_events      : {len(raw_azimuths)}\n")
            f.write(f"bins          : {BINS}\n")
            f.write(f"pdf_sum       : {pdf.sum():.8f}\n")
            f.write(f"azimuth_mean  : {np.mean(raw_azimuths):.2f}\n")
            f.write(f"azimuth_std   : {np.std(raw_azimuths):.2f}\n")
        print(f"[OK] Metadata tersimpan -> {meta_path}")

        # Visualisasi
        plot_path = out_dir / f"prior_distribution_polar_{station_id}.png"
        plot_polar_prior_station(pdf, station_id, stn_lat, stn_lon, plot_path, raw_azimuths=raw_azimuths)
        
        priors_built += 1

    print(f"\n" + "=" * 60)
    print(f"  STATION-SPECIFIC PRIORS BUILD COMPLETE")
    print("=" * 60)
    print(f"  Priors built   : {priors_built}/{len(STATION_COORDINATES)} stations")
    print(f"  Output dir     : {out_dir}")
    print(f"  Catalog events : {len(df)}")
    print(f"  Bandwidth      : {bandwidth}deg")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="Build station-specific dynamic priors for V9.1")
    p.add_argument("--catalog", default="2026.csv", help="Path to earthquake catalog")
    p.add_argument("--bandwidth", type=float, default=15.0, help="KDE bandwidth in degrees")
    p.add_argument("--output-dir", default="Bayesian/priors", help="Output directory for priors")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_station_priors(args.catalog, args.bandwidth, args.output_dir)
