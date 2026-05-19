"""
Validate V10 Multi-Station Graph Dataset
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_FILE = 'dataset_v10_train_val_graphs.h5'
PLOT_DIR = 'plots'

def validate_v10_dataset():
    """Validate the V10 dataset."""
    print("="*70)
    print("VALIDATING V10 MULTI-STATION GRAPH DATASET")
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
                    attr_value = group.attrs[attr_name]
                    if isinstance(attr_value, np.ndarray):
                        print(f"    - {attr_name}: {list(attr_value)}")
                    else:
                        print(f"    - {attr_name}: {attr_value}")
        
        # Global attributes
        if f.attrs:
            print(f"\n📋 Global Attributes:")
            for attr_name in f.attrs:
                print(f"  - {attr_name}: {f.attrs[attr_name]}")
        
        # Validate train group
        if 'train' in f:
            print("\n" + "="*70)
            print("TRAIN SET ANALYSIS")
            print("="*70)
            
            train_group = f['train']
            
            # Tensor shape
            tensors = train_group['tensors']
            print(f"\n✓ Tensors shape: {tensors.shape}")
            print(f"  Format: (num_days={tensors.shape[0]}, num_stations={tensors.shape[1]}, "
                  f"channels={tensors.shape[2]}, height={tensors.shape[3]}, width={tensors.shape[4]})")
            
            # Dates
            dates_raw = train_group['dates'][:]
            if isinstance(dates_raw[0], bytes):
                dates = [pd.to_datetime(d.decode('utf-8')) for d in dates_raw]
            else:
                dates = [pd.to_datetime(d) for d in dates_raw]
            print(f"\n✓ Date range: {min(dates).date()} to {max(dates).date()}")
            print(f"  Total days: {len(dates)}")
            
            # Labels
            events = train_group['label_event'][:]
            mags = train_group['label_mag'][:]
            azms = train_group['label_azm'][:]
            
            print(f"\n✓ Labels:")
            print(f"  Events: {np.sum(events)} events, {len(events) - np.sum(events)} background")
            print(f"  Event ratio: {np.sum(events)/len(events)*100:.1f}%")
            print(f"  Magnitudes: min={mags[mags > 0].min():.2f}, max={mags.max():.2f}, mean={mags[mags > 0].mean():.2f}")
            print(f"  Azimuths shape: {azms.shape}")
            
            # Cosmic features
            cosmic = train_group['cosmic_features'][:]
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
        
        # Validate val group
        if 'val' in f:
            print("\n" + "="*70)
            print("VALIDATION SET ANALYSIS")
            print("="*70)
            
            val_group = f['val']
            
            # Tensor shape
            tensors = val_group['tensors']
            print(f"\n✓ Tensors shape: {tensors.shape}")
            print(f"  Format: (num_days={tensors.shape[0]}, num_stations={tensors.shape[1]}, "
                  f"channels={tensors.shape[2]}, height={tensors.shape[3]}, width={tensors.shape[4]})")
            
            # Dates
            dates_raw = val_group['dates'][:]
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
            print(f"  Event ratio: {np.sum(events)/len(events)*100:.1f}%")
            print(f"  Magnitudes: min={mags[mags > 0].min():.2f}, max={mags.max():.2f}, mean={mags[mags > 0].mean():.2f}")
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
            
            # Check for May 2024 storm
            may_2024_start = pd.to_datetime('2024-05-01')
            may_2024_end = pd.to_datetime('2024-05-31')
            may_dates = [d for d in dates if may_2024_start <= d <= may_2024_end]
            if len(may_dates) > 0:
                may_indices = [i for i, d in enumerate(dates) if may_2024_start <= d <= may_2024_end]
                may_kp = cosmic[may_indices, 0]
                print(f"\n✓ May 2024 Storm Data:")
                print(f"  Days in May 2024: {len(may_dates)}")
                print(f"  Kp range: {may_kp.min():.2f} to {may_kp.max():.2f}")
                print(f"  Days with Kp >= 8.0: {np.sum(may_kp >= 8.0)}")
            
            # Create visualization
            create_v10_visualization(f)
    
    print("\n" + "="*70)
    print("✅ VALIDATION COMPLETE")
    print("="*70)


def create_v10_visualization(f):
    """Create visualization for V10 dataset."""
    print("\n📊 Creating visualization...")
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Train temporal distribution
    if 'train' in f:
        ax1 = fig.add_subplot(gs[0, 0])
        train_dates = f['train']['dates'][:]
        if isinstance(train_dates[0], bytes):
            train_dates = [pd.to_datetime(d.decode('utf-8')) for d in train_dates]
        else:
            train_dates = [pd.to_datetime(d) for d in train_dates]
        
        train_events = f['train']['label_event'][:]
        event_dates = [train_dates[i] for i, e in enumerate(train_events) if e == 1]
        noise_dates = [train_dates[i] for i, e in enumerate(train_events) if e == 0]
        
        ax1.scatter(event_dates, [1]*len(event_dates), c='red', label='Events', alpha=0.6, s=20)
        ax1.scatter(noise_dates, [0]*len(noise_dates), c='blue', label='Noise', alpha=0.6, s=20)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Label')
        ax1.set_title('Train Set: Temporal Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Val temporal distribution
    if 'val' in f:
        ax2 = fig.add_subplot(gs[0, 1])
        val_dates = f['val']['dates'][:]
        if isinstance(val_dates[0], bytes):
            val_dates = [pd.to_datetime(d.decode('utf-8')) for d in val_dates]
        else:
            val_dates = [pd.to_datetime(d) for d in val_dates]
        
        val_events = f['val']['label_event'][:]
        event_dates = [val_dates[i] for i, e in enumerate(val_events) if e == 1]
        noise_dates = [val_dates[i] for i, e in enumerate(val_events) if e == 0]
        
        ax2.scatter(event_dates, [1]*len(event_dates), c='red', label='Events', alpha=0.6, s=20)
        ax2.scatter(noise_dates, [0]*len(noise_dates), c='blue', label='Noise', alpha=0.6, s=20)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Label')
        ax2.set_title('Val Set: Temporal Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Event distribution comparison
    ax3 = fig.add_subplot(gs[0, 2])
    if 'train' in f and 'val' in f:
        train_events = f['train']['label_event'][:]
        val_events = f['val']['label_event'][:]
        
        x = ['Train', 'Val']
        events = [np.sum(train_events), np.sum(val_events)]
        noise = [len(train_events) - np.sum(train_events), len(val_events) - np.sum(val_events)]
        
        width = 0.35
        ax3.bar([0], events[0], width, label='Events', color='red', alpha=0.7)
        ax3.bar([0], noise[0], width, bottom=events[0], label='Noise', color='blue', alpha=0.7)
        ax3.bar([1], events[1], width, color='red', alpha=0.7)
        ax3.bar([1], noise[1], width, bottom=events[1], color='blue', alpha=0.7)
        
        ax3.set_ylabel('Count')
        ax3.set_title('Event vs Noise Distribution')
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(x)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Train magnitude distribution
    if 'train' in f:
        ax4 = fig.add_subplot(gs[1, 0])
        mags = f['train']['label_mag'][:]
        valid_mags = mags[mags > 0]
        if len(valid_mags) > 0:
            ax4.hist(valid_mags, bins=20, color='purple', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Magnitude')
            ax4.set_ylabel('Count')
            ax4.set_title('Train Set: Magnitude Distribution')
            ax4.grid(True, alpha=0.3)
    
    # Plot 5: Val magnitude distribution
    if 'val' in f:
        ax5 = fig.add_subplot(gs[1, 1])
        mags = f['val']['label_mag'][:]
        valid_mags = mags[mags > 0]
        if len(valid_mags) > 0:
            ax5.hist(valid_mags, bins=20, color='purple', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Magnitude')
            ax5.set_ylabel('Count')
            ax5.set_title('Val Set: Magnitude Distribution')
            ax5.grid(True, alpha=0.3)
    
    # Plot 6: Space weather (Kp vs Dst)
    ax6 = fig.add_subplot(gs[1, 2])
    if 'train' in f and 'val' in f:
        train_cosmic = f['train']['cosmic_features'][:]
        val_cosmic = f['val']['cosmic_features'][:]
        
        ax6.scatter(train_cosmic[:, 0], train_cosmic[:, 1], alpha=0.5, s=20, label='Train', c='blue')
        ax6.scatter(val_cosmic[:, 0], val_cosmic[:, 1], alpha=0.5, s=20, label='Val', c='red')
        ax6.set_xlabel('Kp Index')
        ax6.set_ylabel('Dst Index')
        ax6.set_title('Space Weather Features')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: Sample scalogram (Train)
    if 'train' in f:
        ax7 = fig.add_subplot(gs[2, 0])
        tensors = f['train']['tensors']
        sample = tensors[0, 0, 0]  # First day, first station, first channel
        im = ax7.imshow(sample, aspect='auto', cmap='viridis')
        ax7.set_xlabel('Time')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Train: Sample Scalogram (Day 0, Station 0, Ch 0)')
        plt.colorbar(im, ax=ax7)
    
    # Plot 8: Sample scalogram (Val)
    if 'val' in f:
        ax8 = fig.add_subplot(gs[2, 1])
        tensors = f['val']['tensors']
        sample = tensors[0, 0, 0]  # First day, first station, first channel
        im = ax8.imshow(sample, aspect='auto', cmap='viridis')
        ax8.set_xlabel('Time')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Val: Sample Scalogram (Day 0, Station 0, Ch 0)')
        plt.colorbar(im, ax=ax8)
    
    # Plot 9: Station coverage
    ax9 = fig.add_subplot(gs[2, 2])
    if 'train' in f and 'val' in f:
        train_stations = f['train'].attrs['stations']
        val_stations = f['val'].attrs['stations']
        
        # Convert bytes to strings if needed
        if isinstance(train_stations[0], bytes):
            train_stations = [s.decode('utf-8') for s in train_stations]
        if isinstance(val_stations[0], bytes):
            val_stations = [s.decode('utf-8') for s in val_stations]
        
        all_stations = sorted(set(train_stations) | set(val_stations))
        train_mask = [1 if s in train_stations else 0 for s in all_stations]
        val_mask = [1 if s in val_stations else 0 for s in all_stations]
        
        x = np.arange(len(all_stations))
        width = 0.35
        
        ax9.bar(x - width/2, train_mask, width, label='Train', alpha=0.7)
        ax9.bar(x + width/2, val_mask, width, label='Val', alpha=0.7)
        ax9.set_xlabel('Station')
        ax9.set_ylabel('Present (1) / Absent (0)')
        ax9.set_title('Station Coverage')
        ax9.set_xticks(x)
        ax9.set_xticklabels(all_stations, rotation=45, ha='right')
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('V10 Multi-Station Graph Dataset - Train & Val Sets', 
                fontsize=16, fontweight='bold')
    
    output_file = os.path.join(PLOT_DIR, 'v10_dataset_validation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to: {output_file}")


if __name__ == '__main__':
    validate_v10_dataset()
