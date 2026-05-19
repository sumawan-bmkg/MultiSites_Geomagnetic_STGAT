import h5py
import numpy as np
import pandas as pd
from datetime import datetime

HDF5_FILE = 'dataset_v13_train_val_M5_patched.h5'

def analyze_dataset():
    with h5py.File(HDF5_FILE, 'r') as f:
        stats_list = []
        for split in ['train', 'val']:
            if split not in f: continue
            g = f[split]
            dates_raw = g['dates'][:]
            dates = [d.decode() if isinstance(d, bytes) else d for d in dates_raw]
            
            # Events and Magnitudes
            events = g['label_event'][:]
            mags = g['label_mag'][:]
            event_mags = mags[events == 1]
            
            # Cosmic Features (Index 0: Kp, Index 1: Dst)
            cosmic = g['cosmic_features'][:]
            kp = cosmic[:, 0]
            dst = cosmic[:, 1]
            
            # Station coverage
            stations = g.attrs.get('stations', [])
            
            # Tensors (sampling only first and last for speed)
            tensors = g['tensors']
            first_t = tensors[0]
            last_t = tensors[-1]
            
            stats = {
                'Split': split.capitalize(),
                'Samples': len(dates),
                'Start Date': min(dates),
                'End Date': max(dates),
                'Events (Mw > 5.0)': int(np.sum(events)),
                'Noise': len(dates) - int(np.sum(events)),
                'Min Mw': float(np.min(event_mags)) if len(event_mags) > 0 else 0,
                'Max Mw': float(np.max(event_mags)) if len(event_mags) > 0 else 0,
                'Mean Mw': float(np.mean(event_mags)) if len(event_mags) > 0 else 0,
                'Mean Kp': float(np.mean(kp)),
                'Mean Dst': float(np.mean(dst)),
                'Stations': len(stations),
                'Non-Zero Content (Start)': np.count_nonzero(first_t) / first_t.size * 100,
                'Non-Zero Content (End)': np.count_nonzero(last_t) / last_t.size * 100
            }
            stats_list.append(stats)
            
        df = pd.DataFrame(stats_list)
        print(df.to_string(index=False))

if __name__ == '__main__':
    analyze_dataset()
