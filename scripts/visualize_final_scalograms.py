import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

HDF5_FILE = 'dataset_v13_train_val_M5_patched.h5'
OUTPUT_DIR = 'plots/visual_check'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_samples():
    with h5py.File(HDF5_FILE, 'r') as f:
        # Get train data
        group = f['train']
        tensors = group['tensors']
        dates = group['dates']
        events = group['label_event'][:]
        stations = group.attrs.get('stations', [])
        
        # Find an event day
        event_indices = np.where(events == 1)[0]
        if len(event_indices) == 0:
            print("No event found in train set.")
            return
            
        # Select first event day
        idx = event_indices[0]
        date_str = dates[idx].decode() if isinstance(dates[idx], bytes) else dates[idx]
        
        # Select a station with non-zero data
        # Check first few stations
        for s_idx in range(len(stations)):
            sample = tensors[idx, s_idx] # (3, 128, 1440)
            if np.any(sample > 0):
                stn_name = stations[s_idx].decode() if isinstance(stations[s_idx], bytes) else stations[s_idx]
                print(f"Plotting {date_str} at station {stn_name}")
                
                fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
                channels = ['H-component', 'D-component', 'Z-component']
                
                for c in range(3):
                    ax = axes[c]
                    im = ax.imshow(sample[c], aspect='auto', origin='lower', cmap='jet')
                    ax.set_title(f"{stn_name} - {channels[c]}")
                    ax.set_ylabel("Frequency Bin (Pc3)")
                    fig.colorbar(im, ax=ax)
                
                axes[2].set_xlabel("Time (Minutes of Day)")
                plt.suptitle(f"Real Geomagnetic Scalogram (Pc3 Filtered)\nDate: {date_str} | Station: {stn_name}", fontsize=14)
                
                save_path = os.path.join(OUTPUT_DIR, f"scalogram_{date_str}_{stn_name}.png")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(save_path)
                print(f"Saved to {save_path}")
                break

if __name__ == '__main__':
    visualize_samples()
