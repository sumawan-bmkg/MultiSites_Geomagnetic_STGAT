"""
Validate the new multi-station graph dataset (v9).
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_FILE = 'scalogram_v9_multistation_graph.h5'
PLOT_DIR = 'plots'

def validate_v9_dataset():
    """Validate the transformed dataset."""
    print("="*70)
    print("VALIDATING MULTI-STATION GRAPH DATASET (V9)")
    print("="*70)
    
    if not os.path.exists(OUTPUT_FILE):
        print(f"\n❌ File not found: {OUTPUT_FILE}")
        return
    
    with h5py.File(OUTPUT_FILE, 'r') as f:
        print(f"\n✓ File opened: {OUTPUT_FILE}")
        print(f"  File size: {os.path.getsize(OUTPUT_FILE) / (1024**2):.2f} MB")
        
        # Print structure
        print("\nDataset Structure:")
        print("-" * 70)
        
        for group_name in f.keys():
            print(f"\n📁 Group: {group_name}")
            group = f[group_name]
            
            for key in group.keys():
                dataset = group[key]
                print(f"  📊 {key}: {dataset.shape}, dtype={dataset.dtype}")
            
            # Print attributes
            if group.attrs:
                print(f"  Attributes:")
                for attr_name in group.attrs:
                    print(f"    - {attr_name}: {group.attrs[attr_name]}")
        
        # Global attributes
        if f.attrs:
            print(f"\n📋 Global Attributes:")
            for attr_name in f.attrs:
                print(f"  - {attr_name}: {f.attrs[attr_name]}")
        
        # Validate val group (only one with data)
        if 'val' in f:
            print("\n" + "="*70)
            print("VALIDATION GROUP ANALYSIS")
            print("="*70)
            
            val_group = f['val']
            
            # Tensor shape
            tensors = val_group['tensors']
            print(f"\n✓ Tensors shape: {tensors.shape}")
            print(f"  Format: (num_days={tensors.shape[0]}, num_stations={tensors.shape[1]}, "
                  f"channels={tensors.shape[2]}, height={tensors.shape[3]}, width={tensors.shape[4]})")
            
            # Dates
            dates_raw = val_group['dates'][:]
            # Decode bytes to string if necessary
            if isinstance(dates_raw[0], bytes):
                dates = [pd.to_datetime(d.decode('utf-8')) for d in dates_raw]
            else:
                dates = [pd.to_datetime(d) for d in dates_raw]
            print(f"\n✓ Date range: {min(dates).date()} to {max(dates).date()}")
            print(f"  Total days: {len(dates)}")
            
            # Labels
            events = val_group['label_event'][:]
            mags = val_group['label_mag'][:]
            azms = val_group['label_azm'][:]
            
            print(f"\n✓ Labels:")
            print(f"  Events: {np.sum(events)} events, {len(events) - np.sum(events)} background")
            print(f"  Magnitudes: min={mags.min():.2f}, max={mags.max():.2f}, mean={mags.mean():.2f}")
            print(f"  Azimuths shape: {azms.shape}")
            
            # Cosmic features
            cosmic = val_group['cosmic_features'][:]
            print(f"\n✓ Cosmic features shape: {cosmic.shape}")
            print(f"  Kp range: {cosmic[:, 0].min():.2f} to {cosmic[:, 0].max():.2f}")
            print(f"  Dst range: {cosmic[:, 1].min():.2f} to {cosmic[:, 1].max():.2f}")
            
            # Check for NaN
            print(f"\n✓ Data quality:")
            print(f"  NaN in tensors (sample 10): {np.isnan(tensors[:10]).sum()}")
            print(f"  NaN in events: {np.isnan(events).sum()}")
            print(f"  NaN in mags: {np.isnan(mags).sum()}")
            print(f"  NaN in azms: {np.isnan(azms).sum()}")
            print(f"  NaN in cosmic: {np.isnan(cosmic).sum()}")
            
            # Create visualization
            create_v9_visualization(val_group, dates)
    
    print("\n" + "="*70)
    print("✅ VALIDATION COMPLETE")
    print("="*70)


def create_v9_visualization(val_group, dates):
    """Create visualization for v9 dataset."""
    print("\n📊 Creating visualization...")
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Temporal distribution
    ax1 = fig.add_subplot(gs[0, 0])
    dates_series = pd.Series(dates)
    daily_counts = dates_series.value_counts().sort_index()
    ax1.plot(range(len(daily_counts)), daily_counts.values, marker='o', linewidth=2)
    ax1.set_xlabel('Day Index')
    ax1.set_ylabel('Samples per Day')
    ax1.set_title('Temporal Distribution (Val Set)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Event distribution
    ax2 = fig.add_subplot(gs[0, 1])
    events = val_group['label_event'][:]
    unique, counts = np.unique(events, return_counts=True)
    ax2.bar(unique, counts, color=['red', 'green'])
    ax2.set_xlabel('Label')
    ax2.set_ylabel('Count')
    ax2.set_title('Event Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Magnitude distribution
    ax3 = fig.add_subplot(gs[0, 2])
    mags = val_group['label_mag'][:]
    valid_mags = mags[mags > 0]
    if len(valid_mags) > 0:
        ax3.hist(valid_mags, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Magnitude')
        ax3.set_ylabel('Count')
        ax3.set_title('Magnitude Distribution (events only)')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Azimuth heatmap (sample)
    ax4 = fig.add_subplot(gs[1, 0])
    azms = val_group['label_azm'][:10]  # First 10 days
    im = ax4.imshow(azms.T, aspect='auto', cmap='twilight', vmin=0, vmax=360)
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Station Index')
    ax4.set_title('Azimuth per Station (sample 10 days)')
    plt.colorbar(im, ax=ax4, label='Azimuth (degrees)')
    
    # Plot 5: Cosmic features
    ax5 = fig.add_subplot(gs[1, 1])
    cosmic = val_group['cosmic_features'][:]
    ax5.scatter(cosmic[:, 0], cosmic[:, 1], alpha=0.6, s=20)
    ax5.set_xlabel('Kp Index')
    ax5.set_ylabel('Dst Index')
    ax5.set_title('Space Weather Features')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Sample scalogram
    ax6 = fig.add_subplot(gs[1, 2])
    tensors = val_group['tensors']
    sample = tensors[0, 0, 0]  # First day, first station, first channel
    im = ax6.imshow(sample, aspect='auto', cmap='viridis')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Sample Scalogram (Day 0, Station 0, Ch 0)')
    plt.colorbar(im, ax=ax6)
    
    plt.suptitle('Multi-Station Graph Dataset (V9) - Validation Set', 
                fontsize=14, fontweight='bold')
    
    output_file = os.path.join(PLOT_DIR, 'v9_multistation_validation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to: {output_file}")


if __name__ == '__main__':
    validate_v9_dataset()
