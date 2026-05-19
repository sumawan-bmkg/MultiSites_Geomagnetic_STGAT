import h5py
import numpy as np

def inspect_h5(path):
    print(f"Inspecting {path}...")
    with h5py.File(path, 'r') as f:
        for split in f.keys():
            print(f"\nSplit: {split}")
            group = f[split]
            print(f"  Keys: {list(group.keys())}")
            for attr in group.attrs:
                val = group.attrs[attr]
                if isinstance(val, np.ndarray):
                    print(f"  Attr {attr}: {val.shape}")
                else:
                    print(f"  Attr {attr}: {val}")
            
            if 'dates' in group:
                print(f"  Dates range: {group['dates'][0]} to {group['dates'][-1]}")

if __name__ == "__main__":
    inspect_h5('dataset_v13_train_val_M5_patched.h5')
    inspect_h5('dataset_v13_blindtest_M5_patched.h5')
