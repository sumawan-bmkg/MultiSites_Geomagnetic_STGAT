"""
ANTIGRAVITY Project - Scalogram Dataset Validation
===================================================

Validasi dataset scalogram HDF5 dengan struktur:
- train/tensors: (9456, 3, 128, 1440)
- train/label_event, label_mag, label_azm
- train/cosmic_features
- train/meta

Author: ANTIGRAVITY Team
Date: 2026-05-02
"""

import h5py
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configuration
HDF5_FILE = 'scalogram_v8_true_negatives.h5'
OUTPUT_DIR = 'plots'
EXPECTED_NUM_STATIONS = 24  # Expected from metadata

# Expected date ranges
TRAIN_END_DATE = '2023-12-31'
VAL_START_DATE = '2024-01-01'
VAL_END_DATE = '2025-03-31'
TEST_START_DATE = '2026-01-01'


class ValidationReport:
    """Class untuk menyimpan hasil validasi."""
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'info': []
        }
    
    def add(self, status: str, test_name: str, message: str = ""):
        """Add test result."""
        self.results[status].append((test_name, message))
        
        status_symbol = {
            'passed': '[PASSED]',
            'failed': '[FAILED]',
            'warnings': '[WARNING]',
            'info': '[INFO]'
        }
        
        print(f"{status_symbol[status]} {test_name}")
        if message:
            indent = "         " if status == 'passed' else "          "
            print(f"{indent}{message}")
    
    def summary(self):
        """Print summary."""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"✅ Passed:   {len(self.results['passed'])}")
        print(f"❌ Failed:   {len(self.results['failed'])}")
        print(f"⚠️  Warnings: {len(self.results['warnings'])}")
        print(f"ℹ️  Info:     {len(self.results['info'])}")
        
        if self.results['failed']:
            print("\n" + "="*70)
            print("FAILED TESTS:")
            print("="*70)
            for name, msg in self.results['failed']:
                print(f"\n❌ {name}:")
                print(f"   {msg}")
        
        if self.results['warnings']:
            print("\n" + "="*70)
            print("WARNINGS:")
            print("="*70)
            for name, msg in self.results['warnings']:
                print(f"\n⚠️  {name}:")
                print(f"   {msg}")
        
        print("\n" + "="*70)
        if len(self.results['failed']) == 0:
            print("✅ ALL CRITICAL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED - Please review!")
        print("="*70)
        
        return len(self.results['failed']) == 0


def extract_dates_from_meta(meta_array):
    """Extract dates from metadata filenames."""
    dates = []
    stations = []
    
    for meta in meta_array:
        # Decode if bytes
        if isinstance(meta, bytes):
            meta = meta.decode('utf-8')
        
        # Extract date from filename like 'event_ALR_20250824.npy'
        parts = meta.split('_')
        if len(parts) >= 3:
            station = parts[1]
            date_str = parts[2].replace('.npy', '')
            
            # Parse date: YYYYMMDD
            if len(date_str) == 8:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                date = pd.Timestamp(year=year, month=month, day=day)
                dates.append(date)
                stations.append(station)
    
    return dates, stations


def validate_dataset_structure(report: ValidationReport):
    """UJI 1: Validasi Struktur Dataset."""
    print("\n" + "="*70)
    print("UJI 1: VALIDASI STRUKTUR DATASET")
    print("="*70)
    
    try:
        with h5py.File(HDF5_FILE, 'r') as f:
            # Check groups
            groups = list(f.keys())
            report.add('info', 'Dataset Groups', f"Found groups: {groups}")
            
            # Check train group
            if 'train' in f:
                train_group = f['train']
                train_keys = list(train_group.keys())
                report.add('passed', 'Train Group', f"Found keys: {train_keys}")
                
                # Check tensors
                if 'tensors' in train_group:
                    tensors_shape = train_group['tensors'].shape
                    report.add('info', 'Train Tensors Shape', str(tensors_shape))
                    
                    # Check dimensions
                    # Expected: (samples, channels, height, width)
                    if len(tensors_shape) == 4:
                        n_samples, n_channels, height, width = tensors_shape
                        report.add('passed', 'UJI 1.1 - Tensor Dimensions',
                            f"Shape: ({n_samples}, {n_channels}, {height}, {width})")
                        
                        # Note: This format is per-station, not multi-station
                        report.add('warnings', 'Multi-Station Format',
                            f"Data appears to be per-station format (3 channels). "
                            f"Expected multi-station format would be (samples, 24, channels, h, w)")
                    else:
                        report.add('failed', 'UJI 1.1 - Tensor Dimensions',
                            f"Unexpected shape: {tensors_shape}")
                
                # Check labels
                label_keys = ['label_event', 'label_mag', 'label_azm']
                for key in label_keys:
                    if key in train_group:
                        shape = train_group[key].shape
                        report.add('passed', f'Label {key}', f"Shape: {shape}")
                    else:
                        report.add('failed', f'Label {key}', "Not found")
            
            # Check val group
            if 'val' in f:
                val_group = f['val']
                val_keys = list(val_group.keys())
                report.add('passed', 'Val Group', f"Found keys: {val_keys}")
                
                if 'tensors' in val_group:
                    tensors_shape = val_group['tensors'].shape
                    report.add('info', 'Val Tensors Shape', str(tensors_shape))
    
    except Exception as e:
        report.add('failed', 'Structure Validation', str(e))


def validate_spatial_integrity(report: ValidationReport):
    """UJI 1: Validasi Integritas Spasial."""
    print("\n" + "="*70)
    print("UJI 1: VALIDASI INTEGRITAS SPASIAL (NaN Check)")
    print("="*70)
    
    try:
        with h5py.File(HDF5_FILE, 'r') as f:
            # Check train tensors for NaN
            if 'train' in f and 'tensors' in f['train']:
                tensors = f['train']['tensors']
                
                # Sample first 100 entries
                sample_size = min(100, tensors.shape[0])
                sample_data = tensors[:sample_size]
                
                nan_count = np.isnan(sample_data).sum()
                inf_count = np.isinf(sample_data).sum()
                
                if nan_count == 0 and inf_count == 0:
                    report.add('passed', 'UJI 1.2 - NaN/Inf Check',
                        f"No NaN or Inf values in {sample_size} sampled tensors")
                else:
                    report.add('failed', 'UJI 1.2 - NaN/Inf Check',
                        f"Found {nan_count} NaN and {inf_count} Inf values")
    
    except Exception as e:
        report.add('failed', 'Spatial Integrity', str(e))


def validate_chronological_split(report: ValidationReport):
    """UJI 2: Validasi Strict Chronological Split."""
    print("\n" + "="*70)
    print("UJI 2: VALIDASI STRICT CHRONOLOGICAL SPLIT")
    print("="*70)
    
    try:
        with h5py.File(HDF5_FILE, 'r') as f:
            # Extract dates from train metadata
            if 'train' in f and 'meta' in f['train']:
                train_meta = f['train']['meta'][:]
                train_dates, train_stations = extract_dates_from_meta(train_meta)
                
                if train_dates:
                    train_dates_series = pd.Series(train_dates)
                    report.add('info', 'Train Date Range',
                        f"{train_dates_series.min().date()} to {train_dates_series.max().date()}")
                    
                    # Check if all dates <= 2023-12-31
                    train_end = pd.to_datetime(TRAIN_END_DATE)
                    violations = train_dates_series[train_dates_series > train_end]
                    
                    if len(violations) == 0:
                        report.add('passed', 'UJI 2.1 - Train Date Range',
                            f"All {len(train_dates)} samples ≤ {TRAIN_END_DATE}")
                    else:
                        report.add('failed', 'UJI 2.1 - Train Date Range',
                            f"Found {len(violations)} samples > {TRAIN_END_DATE}")
                    
                    # Station distribution
                    station_counts = Counter(train_stations)
                    report.add('info', 'Train Stations',
                        f"Found {len(station_counts)} unique stations: {dict(station_counts)}")
                else:
                    report.add('warnings', 'Train Dates',
                        "Could not extract dates from metadata")
            
            # Extract dates from val metadata
            if 'val' in f and 'meta' in f['val']:
                val_meta = f['val']['meta'][:]
                val_dates, val_stations = extract_dates_from_meta(val_meta)
                
                if val_dates:
                    val_dates_series = pd.Series(val_dates)
                    report.add('info', 'Val Date Range',
                        f"{val_dates_series.min().date()} to {val_dates_series.max().date()}")
                    
                    # Check if dates in [2024-01-01, 2025-03-31]
                    val_start = pd.to_datetime(VAL_START_DATE)
                    val_end = pd.to_datetime(VAL_END_DATE)
                    
                    in_range = val_dates_series[(val_dates_series >= val_start) & 
                                               (val_dates_series <= val_end)]
                    
                    if len(in_range) == len(val_dates_series):
                        report.add('passed', 'UJI 2.2 - Val Date Range',
                            f"All {len(val_dates)} samples in [{VAL_START_DATE}, {VAL_END_DATE}]")
                    else:
                        report.add('warnings', 'UJI 2.2 - Val Date Range',
                            f"Only {len(in_range)}/{len(val_dates)} samples in expected range")
                    
                    # Check for May 2024
                    may_2024 = val_dates_series[(val_dates_series.dt.year == 2024) & 
                                               (val_dates_series.dt.month == 5)]
                    
                    if len(may_2024) > 0:
                        report.add('passed', 'UJI 2.4 - May 2024 Data',
                            f"Found {len(may_2024)} samples from May 2024")
                    else:
                        report.add('warnings', 'UJI 2.4 - May 2024 Data',
                            "No May 2024 samples found")
                    
                    # Station distribution
                    station_counts = Counter(val_stations)
                    report.add('info', 'Val Stations',
                        f"Found {len(station_counts)} unique stations: {dict(station_counts)}")
    
    except Exception as e:
        report.add('failed', 'Chronological Split', str(e))


def validate_labels(report: ValidationReport):
    """UJI 3: Validasi Target DPINN & Multi-Task."""
    print("\n" + "="*70)
    print("UJI 3: VALIDASI TARGET DPINN & MULTI-TASK")
    print("="*70)
    
    try:
        with h5py.File(HDF5_FILE, 'r') as f:
            if 'train' in f:
                train_group = f['train']
                
                # Check label_event
                if 'label_event' in train_group:
                    label_event = train_group['label_event'][:]
                    unique_events = np.unique(label_event)
                    event_counts = Counter(label_event)
                    
                    report.add('info', 'Event Labels',
                        f"Unique values: {unique_events}, Counts: {dict(event_counts)}")
                    
                    if set(unique_events).issubset({0, 1}):
                        report.add('passed', 'UJI 3.1a - Event Labels Binary',
                            f"Labels are binary (0/1)")
                    else:
                        report.add('failed', 'UJI 3.1a - Event Labels Binary',
                            f"Labels contain non-binary values: {unique_events}")
                
                # Check label_mag
                if 'label_mag' in train_group:
                    label_mag = train_group['label_mag'][:]
                    
                    # Check for NaN
                    nan_count = np.isnan(label_mag).sum()
                    if nan_count == 0:
                        report.add('passed', 'UJI 3.1b - Magnitude NaN Check',
                            "No NaN values in magnitude labels")
                    else:
                        report.add('failed', 'UJI 3.1b - Magnitude NaN Check',
                            f"Found {nan_count} NaN values")
                    
                    # Check range
                    valid_mag = label_mag[~np.isnan(label_mag)]
                    if len(valid_mag) > 0:
                        mag_min, mag_max = valid_mag.min(), valid_mag.max()
                        report.add('info', 'Magnitude Range',
                            f"Min: {mag_min:.2f}, Max: {mag_max:.2f}, Mean: {valid_mag.mean():.2f}")
                        
                        if mag_min > 0:
                            report.add('passed', 'UJI 3.1c - Magnitude Positive',
                                "All magnitudes > 0")
                        else:
                            report.add('failed', 'UJI 3.1c - Magnitude Positive',
                                f"Found magnitudes ≤ 0: min = {mag_min}")
                
                # Check label_azm
                if 'label_azm' in train_group:
                    label_azm = train_group['label_azm'][:]
                    
                    # Check for NaN
                    nan_count = np.isnan(label_azm).sum()
                    if nan_count == 0:
                        report.add('passed', 'UJI 3.1d - Azimuth NaN Check',
                            "No NaN values in azimuth labels")
                    else:
                        report.add('failed', 'UJI 3.1d - Azimuth NaN Check',
                            f"Found {nan_count} NaN values")
                    
                    # Check range [0, 360]
                    valid_azm = label_azm[~np.isnan(label_azm)]
                    if len(valid_azm) > 0:
                        azm_min, azm_max = valid_azm.min(), valid_azm.max()
                        report.add('info', 'Azimuth Range',
                            f"Min: {azm_min:.2f}, Max: {azm_max:.2f}")
                        
                        if azm_min >= 0 and azm_max <= 360:
                            report.add('passed', 'UJI 3.1e - Azimuth Range',
                                "All azimuths in [0, 360]")
                        else:
                            report.add('warnings', 'UJI 3.1e - Azimuth Range',
                                f"Some azimuths outside [0, 360]: [{azm_min:.2f}, {azm_max:.2f}]")
                
                # Check cosmic_features
                if 'cosmic_features' in train_group:
                    cosmic = train_group['cosmic_features'][:]
                    report.add('info', 'Cosmic Features Shape', str(cosmic.shape))
                    
                    # Check for NaN
                    nan_count = np.isnan(cosmic).sum()
                    if nan_count == 0:
                        report.add('passed', 'UJI 3.2 - Space Weather NaN Check',
                            "No NaN values in cosmic features")
                    else:
                        report.add('warnings', 'UJI 3.2 - Space Weather NaN Check',
                            f"Found {nan_count} NaN values in cosmic features")
    
    except Exception as e:
        report.add('failed', 'Label Validation', str(e))


def create_visualizations(report: ValidationReport):
    """Create comprehensive visualizations."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        with h5py.File(HDF5_FILE, 'r') as f:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Plot 1: Dataset sizes
            ax1 = fig.add_subplot(gs[0, 0])
            train_size = f['train']['tensors'].shape[0] if 'train' in f else 0
            val_size = f['val']['tensors'].shape[0] if 'val' in f else 0
            
            ax1.bar(['Train', 'Val'], [train_size, val_size], color=['steelblue', 'orange'])
            ax1.set_ylabel('Sample Count')
            ax1.set_title('Dataset Split Sizes')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Label distribution (train)
            ax2 = fig.add_subplot(gs[0, 1])
            if 'train' in f and 'label_event' in f['train']:
                labels = f['train']['label_event'][:]
                unique, counts = np.unique(labels, return_counts=True)
                ax2.bar(unique, counts, color=['red', 'green'])
                ax2.set_xlabel('Label')
                ax2.set_ylabel('Count')
                ax2.set_title('Train: Event Label Distribution')
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Magnitude distribution
            ax3 = fig.add_subplot(gs[0, 2])
            if 'train' in f and 'label_mag' in f['train']:
                mags = f['train']['label_mag'][:]
                valid_mags = mags[~np.isnan(mags)]
                ax3.hist(valid_mags, bins=30, color='purple', alpha=0.7, edgecolor='black')
                ax3.set_xlabel('Magnitude')
                ax3.set_ylabel('Count')
                ax3.set_title('Train: Magnitude Distribution')
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Azimuth distribution
            ax4 = fig.add_subplot(gs[1, 0])
            if 'train' in f and 'label_azm' in f['train']:
                azms = f['train']['label_azm'][:]
                valid_azms = azms[~np.isnan(azms)]
                ax4.hist(valid_azms, bins=36, color='teal', alpha=0.7, edgecolor='black')
                ax4.set_xlabel('Azimuth (degrees)')
                ax4.set_ylabel('Count')
                ax4.set_title('Train: Azimuth Distribution')
                ax4.grid(True, alpha=0.3)
            
            # Plot 5: Temporal distribution (train)
            ax5 = fig.add_subplot(gs[1, 1])
            if 'train' in f and 'meta' in f['train']:
                meta = f['train']['meta'][:]
                dates, _ = extract_dates_from_meta(meta)
                if dates:
                    dates_series = pd.Series(dates)
                    monthly = dates_series.dt.to_period('M').value_counts().sort_index()
                    ax5.plot(range(len(monthly)), monthly.values, marker='o', linewidth=2)
                    ax5.set_xlabel('Month Index')
                    ax5.set_ylabel('Sample Count')
                    ax5.set_title('Train: Temporal Distribution')
                    ax5.grid(True, alpha=0.3)
            
            # Plot 6: Temporal distribution (val)
            ax6 = fig.add_subplot(gs[1, 2])
            if 'val' in f and 'meta' in f['val']:
                meta = f['val']['meta'][:]
                dates, _ = extract_dates_from_meta(meta)
                if dates:
                    dates_series = pd.Series(dates)
                    monthly = dates_series.dt.to_period('M').value_counts().sort_index()
                    ax6.plot(range(len(monthly)), monthly.values, marker='o', 
                            linewidth=2, color='orange')
                    ax6.set_xlabel('Month Index')
                    ax6.set_ylabel('Sample Count')
                    ax6.set_title('Val: Temporal Distribution')
                    ax6.grid(True, alpha=0.3)
            
            # Plot 7: Station distribution (train)
            ax7 = fig.add_subplot(gs[2, 0])
            if 'train' in f and 'meta' in f['train']:
                meta = f['train']['meta'][:]
                _, stations = extract_dates_from_meta(meta)
                if stations:
                    station_counts = Counter(stations)
                    stations_sorted = sorted(station_counts.items(), key=lambda x: x[1], reverse=True)
                    names, counts = zip(*stations_sorted[:15])  # Top 15
                    ax7.barh(names, counts, color='steelblue')
                    ax7.set_xlabel('Sample Count')
                    ax7.set_title('Train: Top 15 Stations')
                    ax7.grid(True, alpha=0.3)
            
            # Plot 8: Cosmic features distribution
            ax8 = fig.add_subplot(gs[2, 1])
            if 'train' in f and 'cosmic_features' in f['train']:
                cosmic = f['train']['cosmic_features'][:1000]  # Sample 1000
                ax8.scatter(cosmic[:, 0], cosmic[:, 1], alpha=0.5, s=10)
                ax8.set_xlabel('Feature 1 (Kp?)')
                ax8.set_ylabel('Feature 2 (Dst?)')
                ax8.set_title('Train: Cosmic Features (sample)')
                ax8.grid(True, alpha=0.3)
            
            # Plot 9: Sample tensor visualization
            ax9 = fig.add_subplot(gs[2, 2])
            if 'train' in f and 'tensors' in f['train']:
                sample_tensor = f['train']['tensors'][0, 0]  # First sample, first channel
                im = ax9.imshow(sample_tensor, aspect='auto', cmap='viridis')
                ax9.set_xlabel('Time')
                ax9.set_ylabel('Frequency')
                ax9.set_title('Sample Scalogram (Channel 0)')
                plt.colorbar(im, ax=ax9)
            
            plt.suptitle('ANTIGRAVITY Dataset Validation Report', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            output_file = os.path.join(OUTPUT_DIR, 'dataset_validation_report.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            report.add('passed', 'Visualization Created', f"Saved to {output_file}")
            print(f"\n📊 Visualization saved to: {output_file}")
    
    except Exception as e:
        report.add('warnings', 'Visualization', f"Error: {str(e)}")


def main():
    """Main execution."""
    print("="*70)
    print("ANTIGRAVITY - SCALOGRAM DATASET VALIDATION")
    print("="*70)
    print(f"File: {HDF5_FILE}")
    print(f"Started: {datetime.now()}")
    print()
    
    report = ValidationReport()
    
    # Check file exists
    if not os.path.exists(HDF5_FILE):
        print(f"\n❌ ERROR: File not found: {HDF5_FILE}")
        return 1
    
    # Run validations
    validate_dataset_structure(report)
    validate_spatial_integrity(report)
    validate_chronological_split(report)
    validate_labels(report)
    
    # Create visualizations
    create_visualizations(report)
    
    # Summary
    all_passed = report.summary()
    
    print(f"\nCompleted: {datetime.now()}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
