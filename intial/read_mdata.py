#!/usr/bin/env python3
"""
read_mdata.py

Python implementation of MATLAB `read_604rcsv_new.m` binary reader for FRG604RC 1-sec data.

Now computes X and Y components from H & D:
    D is assumed in degrees -> D_rad = deg2rad(D)
    X = H * cos(D_rad)
    Y = H * sin(D_rad)

Usage:
    python read_mdata.py --year 2023 --month 1 --day 1 --stn ALR --path mdata
"""
import os
import argparse
import numpy as np
from datetime import datetime, timedelta


def read_604rcsv_new_python(year, month, day, stn, path):
    # follow MATLAB behavior for two-digit year
    yy = year
    if year > 2000:
        yy = year - 2000

    # Try both compressed and uncompressed files
    filename = os.path.join(path, stn, f"S{yy:02d}{month:02d}{day:02d}.{stn}")
    filename_gz = filename + '.gz'
    
    # Check which file exists
    if os.path.exists(filename_gz):
        import gzip
        with gzip.open(filename_gz, 'rb') as f:
            raw = f.read()
    elif os.path.exists(filename):
        with open(filename, 'rb') as f:
            raw = f.read()
    else:
        raise FileNotFoundError(f"File not found: {filename} or {filename_gz}")

    data = np.frombuffer(raw, dtype=np.uint8)

    # block structure: 144 blocks, each block length = 30 + 17*600 = 10230
    stride = 30 + 17 * 600  # 10230
    if data.size % stride != 0:
        n_blocks = data.size // stride
    else:
        n_blocks = data.size // stride

    blocks = []
    volt_list = []
    for i in range(n_blocks):
        start = i * stride
        block = data[start:start+stride]
        if block.size < stride:
            break
        blocks.append(block)
        # MATLAB uses byte position 28 (1-based) -> index 27 (0-based)
        if block.size > 27:
            volt_list.append(int(block[27]))
        else:
            volt_list.append(0)

    if len(blocks) == 0:
        raise ValueError('No complete blocks found in file')

    # build payload by removing first 30 bytes of each block and concatenating
    payload_parts = [b[30:] for b in blocks]
    payload = np.concatenate(payload_parts)
    n_records = payload.size // 17
    if payload.size != n_records * 17:
        payload = payload[:n_records*17]

    records = payload.reshape((n_records, 17))

    # parse fields (little-endian within each multi-byte field)
    def read_uint24_le(arr):
        return arr[:,0].astype(np.uint32) + (arr[:,1].astype(np.uint32) << 8) + (arr[:,2].astype(np.uint32) << 16)

    H_raw = read_uint24_le(records[:,0:3])
    D_raw = records[:,3].astype(np.uint32) + (records[:,4].astype(np.uint32) << 8) + (records[:,5].astype(np.uint32) << 16)
    Z_raw = records[:,6].astype(np.uint32) + (records[:,7].astype(np.uint32) << 8) + (records[:,8].astype(np.uint32) << 16)

    IX_raw = records[:,9].astype(np.uint32) + (records[:,10].astype(np.uint32) << 8)
    IY_raw = records[:,11].astype(np.uint32) + (records[:,12].astype(np.uint32) << 8)
    TempS_raw = records[:,13].astype(np.uint32) + (records[:,14].astype(np.uint32) << 8)
    TempP_raw = records[:,15].astype(np.uint32) + (records[:,16].astype(np.uint32) << 8)

    # sign correction and scaling (match MATLAB)
    def twos_complement(vals, bits):
        vals_signed = vals.copy().astype(np.int64)
        over = vals >= (1 << (bits-1))
        vals_signed[over] = vals_signed[over] - (1 << bits)
        return vals_signed

    H_signed = twos_complement(H_raw, 24)
    D_signed = twos_complement(D_raw, 24)
    Z_signed = twos_complement(Z_raw, 24)

    IX_signed = twos_complement(IX_raw, 16)
    IY_signed = twos_complement(IY_raw, 16)
    TempS_signed = twos_complement(TempS_raw, 16)
    TempP_signed = twos_complement(TempP_raw, 16)

    H = H_signed.astype(np.float64) * 0.01
    D = D_signed.astype(np.float64) * 0.01
    Z = Z_signed.astype(np.float64) * 0.01

    IX = IX_signed.astype(np.float64) * 0.1
    IY = IY_signed.astype(np.float64) * 0.1
    TempS = TempS_signed.astype(np.float64) * 0.01
    TempP = TempP_signed.astype(np.float64) * 0.01

    # apply invalid thresholds (NaN)
    H[np.abs(H) > 80000] = np.nan
    D[np.abs(D) > 80000] = np.nan
    Z[np.abs(Z) > 80000] = np.nan

    IX[np.abs(IX) > 3000] = np.nan
    IY[np.abs(IY) > 3000] = np.nan

    TempS[np.abs(TempS) > 300] = np.nan
    TempP[np.abs(TempP) > 300] = np.nan

    # Voltage: repeat each block voltage value 600 times to form full-day series
    V1 = np.array(volt_list, dtype=np.float64)
    Voltage = np.repeat(V1, 600)
    if Voltage.size > n_records:
        Voltage = Voltage[:n_records]
    elif Voltage.size < n_records:
        Voltage = np.pad(Voltage, (0, n_records - Voltage.size), 'edge')
    Voltage = Voltage * 0.1
    Voltage[Voltage > 24] = np.nan

    # Time vector: create datetime array for the day with 1-second steps
    start = datetime(year, month, day, 0, 0, 0)
    times = np.array([start + timedelta(seconds=i) for i in range(n_records)])

    # Compute X and Y components (assume D in degrees)
    D_rad = np.deg2rad(D)  # if D in degrees
    X = np.full_like(H, np.nan, dtype=np.float64)
    Y = np.full_like(H, np.nan, dtype=np.float64)
    valid_idx = np.isfinite(H) & np.isfinite(D_rad)
    if np.any(valid_idx):
        X[valid_idx] = H[valid_idx] * np.cos(D_rad[valid_idx])
        Y[valid_idx] = H[valid_idx] * np.sin(D_rad[valid_idx])

    return {
        'H': H,
        'D': D,
        'Z': Z,
        'IX': IX,
        'IY': IY,
        'TempS': TempS,
        'TempP': TempP,
        'Voltage': Voltage,
        'Time': times,
        'X': X,
        'Y': Y,
        'filename': filename,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--month', type=int, required=True)
    p.add_argument('--day', type=int, required=True)
    p.add_argument('--stn', type=str, required=True)
    p.add_argument('--path', type=str, default='.')
    args = p.parse_args()

    out = read_604rcsv_new_python(args.year, args.month, args.day, args.stn, args.path)

    n = out['H'].size
    print(f"Read file: {out['filename']}")
    print(f"Records: {n}")
    print(f"H samples: min={np.nanmin(out['H']):.3f}, max={np.nanmax(out['H']):.3f}, mean={np.nanmean(out['H']):.3f}")
    print(f"Voltage: min={np.nanmin(out['Voltage']):.2f}, max={np.nanmax(out['Voltage']):.2f}")

    # save to npz (now includes X and Y)
    yy = args.year
    if args.year > 2000:
        yy = args.year - 2000
    outname = f"S{yy:02d}{args.month:02d}{args.day:02d}.{args.stn}.npz"
    np.savez(outname,
             H=out['H'], D=out['D'], Z=out['Z'],
             IX=out['IX'], IY=out['IY'],
             TempS=out['TempS'], TempP=out['TempP'],
             Voltage=out['Voltage'],
             X=out['X'], Y=out['Y'],
             Time=out['Time'])
    print(f"Saved arrays to {outname}")


if __name__ == '__main__':
    main()
