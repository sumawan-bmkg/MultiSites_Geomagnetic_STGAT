import h5py
import numpy as np
import os
import sys
from datetime import datetime

HDF5_FILE = 'dataset_v13_train_val_M5_patched.h5'

def validate_dataset():
    print("="*70)
    print("ANTIGRAVITY - SCALOGRAM DATASET VALIDATION (REDUCED)")
    print("="*70)
    print(f"File: {HDF5_FILE}")
    print(f"Time: {datetime.now()}")
    print("-" * 70)

    if not os.path.exists(HDF5_FILE):
        print(f"ERROR: File not found: {HDF5_FILE}")
        return

    with h5py.File(HDF5_FILE, 'r') as f:
        for split in ['train', 'val']:
            if split not in f:
                print(f"ERROR: Split '{split}' missing!")
                continue
            
            group = f[split]
            print(f"\nSPLIT: {split.upper()}")
            
            # Check shapes
            tensors = group['tensors']
            dates = group['dates']
            event = group['label_event']
            dist = group['label_dist']
            azm = group['label_azm']
            cosmic = group['cosmic_features']
            
            print(f"  Tensors shape: {tensors.shape}")
            print(f"  Dates count:   {len(dates)}")
            print(f"  Events:        {int(np.sum(event))} / {len(event)}")
            
            # Check tensor stats (to verify real data)
            # Take a sample from train (2018)
            if split == 'train':
                sample = tensors[0]
                non_zero = np.count_nonzero(sample) / sample.size * 100
                print(f"  Sample 0 Non-zero %: {non_zero:.2f}% (Expect > 50% for real data)")
                print(f"  Sample 0 Mean:       {np.mean(sample):.4f}")
                print(f"  Sample 0 Max:        {np.max(sample):.4f}")
            
            # Check Dst patching in Val
            if split == 'val':
                missing_dst = np.sum(cosmic[:, 1] == 0)
                print(f"  Missing Dst:   {missing_dst} (Expect 0 or very low after patching)")
                
            # Check station metadata
            stations = group.attrs.get('stations', [])
            print(f"  Stations:      {len(stations)}")
            
            # Check for label anomalies
            missing_labels = (event[:] == 1) & (np.all(dist[:] == 0, axis=1))
            print(f"  Missing Labels: {np.sum(missing_labels)}")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    validate_dataset()
