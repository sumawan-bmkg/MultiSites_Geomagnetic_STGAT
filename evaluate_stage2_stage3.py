# evaluate_stage2_stage3.py
# --------------------------------------------------------------
# Skrip ini membaca hasil evaluasi akhir (CSV) yang dihasilkan
# oleh evaluate_resilient_v2.py, menyaring True Positive (TP),
# dan menghitung MAE untuk Stage 2 (Magnitudo) serta Stage 3
# (Azimuth/Lokasi). Semua perhitungan dilakukan secara robust.
# --------------------------------------------------------------

import os
import pandas as pd
import numpy as np

# Path ke CSV yang di-save oleh evaluate_resilient_v2.py
csv_path = os.path.join('doc', 'v2_resilient_blindtest_results.csv')

if not os.path.exists(csv_path):
    print(f"[!] ERROR: File tidak ditemukan: {csv_path}")
    exit(1)

# ------------------------------------------------------------------
# 1. Baca data
# ------------------------------------------------------------------
df = pd.read_csv(csv_path)
print("=== INSPEKSI CSV ===")
print("Kolom yang tersedia:", df.columns.tolist())
print("-" * 50)

# ------------------------------------------------------------------
# 2. Filter True Positive (TP)
#    - y_true == 1  (gempa sebenarnya)
#    - prob >= 0.5  (model yakin)
# ------------------------------------------------------------------
# Filter non-silenced True Positives (true events where the model is not silenced by the physics gate)
df_tp = df[(df['y_true'] == 1) & (df['prob'] > 0.0)].copy()
print(f"Total True Positive (TP) untuk analisis: {len(df_tp)} sampel")

# ------------------------------------------------------------------
# 3. MAE Stage 2 – Magnitudo
# ------------------------------------------------------------------
if {'mag_true', 'mag_pred'}.issubset(df_tp.columns):
    mae_mag = np.mean(np.abs(df_tp['mag_true'] - df_tp['mag_pred']))
    print(f"[Stage 2] Mean Absolute Error (MAE) Magnitudo : {mae_mag:.4f}")
else:
    print("[Stage 2] Kolom magnitudo tidak ditemukan pada CSV.")

# ------------------------------------------------------------------
# 4. MAE Stage 3 – Azimuth (lokalisasi)
#    Menggunakan selisih sirkular terkecil (0-360 derajat)
# ------------------------------------------------------------------
if {'azm_true', 'azm_pred'}.issubset(df_tp.columns):
    diff = np.abs(df_tp['azm_true'] - df_tp['azm_pred'])
    circular_diff = np.minimum(diff, 360 - diff)
    mae_azm = np.mean(circular_diff)
    print(f"[Stage 3] Mean Absolute Error (MAE) Azimuth   : {mae_azm:.4f}°")
else:
    print("[Stage 3] Kolom azimuth tidak ditemukan pada CSV.")

print("=" * 50)
