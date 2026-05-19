import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime

# Configuration
HDF5_FILE = 'dataset_v13_train_val_M5_patched.h5'
OUTPUT_DIR = 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_sanity_check():
    print("="*70)
    print("ANTIGRAVITY - SCALOGRAM SANITY CHECK & MAGNITUDE AUDIT")
    print("="*70)

    if not os.path.exists(HDF5_FILE):
        print(f"ERROR: File not found: {HDF5_FILE}")
        return

    with h5py.File(HDF5_FILE, 'r') as f:
        # TAHAP 1: AUDIT MAGNITUDO
        train = f['train']
        y_event = train['label_event'][:]
        y_mag = train['label_mag'][:]
        dates = train['dates'][:]
        
        event_indices = np.where(y_event == 1)[0]
        noise_indices = np.where(y_event == 0)[0]
        
        print(f"\nTAHAP 1: AUDIT MAGNITUDO")
        print(f"Total Event di Train set: {len(event_indices)}")
        
        first_10_mags = y_mag[event_indices[:10]]
        print(f"10 Nilai Magnitudo Pertama (Event): {list(np.round(first_10_mags, 2))}")
        
        if np.all(first_10_mags == 5.0):
            print("WARNING: Terdeteksi bug pemotongan desimal (truncation). Semua nilai adalah 5.0.")
        else:
            print("SUCCESS: Magnitudo bervariasi. Tidak ada pemotongan desimal.")

        # TAHAP 2: EKSTRAKSI SAMPEL VISUAL
        print(f"\nTAHAP 2: EKSTRAKSI SAMPEL")
        
        # Random Event
        idx_ev = random.choice(event_indices)
        date_ev = dates[idx_ev].decode() if isinstance(dates[idx_ev], bytes) else dates[idx_ev]
        mag_ev = y_mag[idx_ev]
        dist_ev = train['label_dist'][idx_ev] # vector of distances
        
        # Random Noise
        idx_no = random.choice(noise_indices)
        date_no = dates[idx_no].decode() if isinstance(dates[idx_no], bytes) else dates[idx_no]
        
        # Pilih stasiun aktif
        stations = train.attrs.get('stations', [])
        tensors = train['tensors']
        
        # Cari stasiun yang aktif di kedua sampel
        best_stn_idx = 0
        for s in range(24):
            data_ev = tensors[idx_ev, s]
            data_no = tensors[idx_no, s]
            if np.count_nonzero(data_ev) > 1000 and np.count_nonzero(data_no) > 1000:
                best_stn_idx = s
                break
        
        stn_name = stations[best_stn_idx]
        if isinstance(stn_name, bytes): stn_name = stn_name.decode()
        
        print(f"Sample Event: {date_ev} | Mw: {mag_ev:.1f} | Station: {stn_name}")
        print(f"Sample Noise: {date_no} | Station: {stn_name}")
        
        # Ekstrak tensor (3, 128, 1440)
        tensor_ev = tensors[idx_ev, best_stn_idx]
        tensor_no = tensors[idx_no, best_stn_idx]
        
        # TAHAP 3: VISUALISASI MATPLOTLIB
        print(f"\nTAHAP 3: VISUALISASI")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        channels = ['H-component', 'D-component', 'Z-component']
        cm = 'jet'
        
        # Ambil jarak spesifik stasiun
        d_val = dist_ev[best_stn_idx]
        
        # Row 1: Event
        for c in range(3):
            ax = axes[0, c]
            im = ax.imshow(tensor_ev[c], aspect='auto', origin='lower', cmap=cm)
            ax.set_title(f"{channels[c]}")
            if c == 0:
                ax.set_ylabel("Freq Bin\n(EVENT)")
            fig.colorbar(im, ax=ax)
        
        axes[0, 1].set_title(f"PRECURSOR EVENT - {date_ev} | Mw: {mag_ev:.1f} | Dist: {d_val:.1f} km\n{channels[1]}")
        
        # Row 2: Noise
        for c in range(3):
            ax = axes[1, c]
            im = ax.imshow(tensor_no[c], aspect='auto', origin='lower', cmap=cm)
            if c == 0:
                ax.set_ylabel("Freq Bin\n(NOISE)")
            fig.colorbar(im, ax=ax)
            
        axes[1, 1].set_title(f"BACKGROUND NOISE - {date_no}\n{channels[1]}")
        
        plt.suptitle(f"ANTIGRAVITY: Scalogram Sanity Check (Event vs Noise)\nStation: {stn_name}", fontsize=16, fontweight='bold')
        axes[1, 1].set_xlabel("Time (Minutes)")
        
        save_path = os.path.join(OUTPUT_DIR, 'scalogram_sanity_check.png')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=150)
        
        print(f"SUCCESS: Gambar berhasil disimpan di {save_path}")

if __name__ == '__main__':
    run_sanity_check()
