#!/usr/bin/env python3
"""
build_spatial_prior.py  --  MODULE 1: SPATIAL PRIOR GENERATOR
=============================================================
Tujuan:
  Membaca katalog gempa historis, menghitung azimuth geodetik dari
  stasiun referensi ke setiap episenter, lalu membangun distribusi
  probabilitas 1D (360 elemen, sum=1) menggunakan Gaussian KDE.

Output:
  - spatial_prior_1d.pt          : torch.Tensor [360] (prior azimuth)
  - prior_distribution_polar.png : visualisasi distribusi dalam polar plot
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

# ------------------------------------------------------------------------------
# Konstanta
# ------------------------------------------------------------------------------
BINS = 360          # resolusi sudut 1 deg/bin
DEFAULT_BW = 15.0   # bandwidth Gaussian KDE dalam derajat


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
# Visualisasi polar
# ------------------------------------------------------------------------------
def plot_polar_prior(pdf: np.ndarray,
                     stn_lat: float,
                     stn_lon: float,
                     output_path: Path,
                     raw_azimuths: np.ndarray = None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib tidak tersedia -- melewati visualisasi.")
        return

    bins    = len(pdf)
    angles  = np.linspace(0.0, 2 * np.pi, bins, endpoint=False)

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw=dict(projection='polar')
    )
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    ax.fill(angles, pdf, alpha=0.35, color='royalblue', label='KDE Prior')
    ax.plot(angles, pdf, color='royalblue', linewidth=2.0)

    if raw_azimuths is not None and len(raw_azimuths) > 0:
        az_rad = np.radians(raw_azimuths)
        max_r  = pdf.max()
        r_vals = np.full_like(az_rad, max_r * 0.08)
        ax.scatter(az_rad, r_vals, color='tomato', s=10, alpha=0.5,
                   zorder=5, label=f'Raw events (n={len(raw_azimuths)})')

    ax.set_title(
        f"Physics-Informed Spatial Prior -- Station ({stn_lat:.4f}deg, {stn_lon:.4f}deg)",
        pad=18, fontsize=13, fontweight='bold'
    )
    ax.legend(loc='lower right', fontsize=10)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Polar plot tersimpan -> {output_path}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def build_prior(catalog_path: str,
                stn_lat: float,
                stn_lon: float,
                bandwidth: float,
                output_dir: str) -> None:

    import pandas as pd

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

    print(f"[INFO] Menghitung azimuth dari stasiun ({stn_lat:.4f}degN, {stn_lon:.4f}degE) ...")
    azimuths = np.array([
        geodesic_azimuth(stn_lat, stn_lon, row[lat_col], row[lon_col])
        for _, row in df.iterrows()
    ], dtype=np.float32)

    print(f"[INFO] Azimuth stats -- mean={np.mean(azimuths):.1f}deg  std={np.std(azimuths):.1f}deg")

    print(f"[INFO] Membangun Gaussian KDE (bandwidth={bandwidth}deg, bins={BINS}) ...")
    pdf = circular_kde_1d(azimuths, bandwidth_deg=bandwidth, bins=BINS)

    print(f"[OK] PDF valid -- sum={pdf.sum():.6f}")

    prior_tensor = torch.tensor(pdf, dtype=torch.float32)
    tensor_path  = out_dir / "spatial_prior_1d.pt"
    torch.save(prior_tensor, str(tensor_path))
    print(f"[OK] Tensor tersimpan -> {tensor_path}  (shape={prior_tensor.shape})")

    meta_path = out_dir / "prior_metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"catalog       : {cat_path.resolve()}\n")
        f.write(f"station_lat   : {stn_lat}\n")
        f.write(f"station_lon   : {stn_lon}\n")
        f.write(f"bandwidth_deg : {bandwidth}\n")
        f.write(f"n_events      : {len(azimuths)}\n")
        f.write(f"bins          : {BINS}\n")
        f.write(f"pdf_sum       : {pdf.sum():.8f}\n")
    print(f"[OK] Metadata tersimpan -> {meta_path}")

    plot_path = out_dir / "prior_distribution_polar.png"
    plot_polar_prior(pdf, stn_lat, stn_lon, plot_path, raw_azimuths=azimuths)

    print("\n" + "=" * 55)
    print("  SPATIAL PRIOR BUILD COMPLETE")
    print("=" * 55)
    print(f"  Tensor  : {tensor_path}")
    print(f"  Plot    : {plot_path}")
    print(f"  Events  : {len(azimuths)}")
    print(f"  BW Deg  : {bandwidth}deg")
    print("=" * 55)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="2026.csv")
    p.add_argument("--stn-lat", type=float, default=0.8126)
    p.add_argument("--stn-lon", type=float, default=127.3670)
    p.add_argument("--bandwidth", type=float, default=15.0)
    p.add_argument("--output-dir", default=".")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_prior(args.catalog, args.stn_lat, args.stn_lon, args.bandwidth, args.output_dir)
