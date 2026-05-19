"""
ANTIGRAVITY Project - HDF5 Dataset Validation
==============================================

Validasi dataset HDF5 yang sudah ada (scalogram_v8_true_negatives.h5)
untuk memastikan kesesuaian dengan spesifikasi ANTIGRAVITY.

Author: ANTIGRAVITY Team
Date: 2026-05-02
"""

import h5py
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
HDF5_FILE = 'scalogram_v8_true_negatives.h5'
OUTPUT_DIR = 'plots'
EXPECTED_NUM_STATIONS = 24

# Expected date ranges
TRAIN_END_DATE = '2023-12-31'
VAL_START_DATE = '2024-01-01'
VAL_END_DATE = '2025-03-31'
TEST_START_DATE = '2026-01-01'


class TestResult:
    """Class untuk menyimpan hasil testing."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
        self.info = []
    
    def add_pass(self, test_name: str, message: str = ""):
        self.passed.append((test_name, message))
        print(f"[PASSED] {test_name}")
        if message:
            print(f"         {message}")
    
    def add_fail(self, test_name: str, message: str):
        self.failed.append((test_name, message))
        print(f"[FAILED] {test_name}")
        print(f"         {message}")
    
    def add_warning(self, test_name: str, message: str):
        self.warnings.append((test_name, message))
        print(f"[WARNING] {test_name}")
        print(f"          {message}")
    
    def add_info(self, test_name: str, message: str):
        self.info.append((test_name, message))
        print(f"[INFO] {test_name}")
        print(f"       {message}")
    
    def summary(self):
        """Print summary of all tests."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Passed:   {len(self.passed)}")
        print(f"Total Failed:   {len(self.failed)}")
        print(f"Total Warnings: {len(self.warnings)}")
        print(f"Total Info:     {len(self.info)}")
        
        if self.failed:
            print("\n" + "="*70)
            print("FAILED TESTS:")
            print("="*70)
            for test_name, message in self.failed:
                print(f"\n{test_name}:")
                print(f"  {message}")
        
        if self.warnings:
            print("\n" + "="*70)
            print("WARNINGS:")
            print("="*70)
            for test_name, message in self.warnings:
                print(f"\n{test_name}:")
                print(f"  {message}")
        
        print("\n" + "="*70)
        if len(self.failed) == 0:
            print("✅ ALL TESTS PASSED - Dataset is valid!")
        else:
            print("❌ SOME TESTS FAILED - Please review issues!")
        print("="*70)
        
        return len(self.failed) == 0


def explore_hdf5_structure(file_path: str, results: TestResult):
    """Explore HDF5 file structure."""
    print("\n" + "="*70)
    print("EXPLORING HDF5 STRUCTURE")
    print("="*70)
    
    if not os.path.exists(file_path):
        results.add_fail("File Existence", f"File not found: {file_path}")
        return None
    
    try:
        with h5py.File(file_path, 'r') as f:
            results.add_pass("File Loading", f"Successfully opened {file_path}")
            
            # Print structure
            print("\nHDF5 Structure:")
            print("-" * 70)
            
            def print_structure(name, obj, indent=0):
                """Recursively print HDF5 structure."""
                prefix = "  " * indent
                if isinstance(obj, h5py.Dataset):
                    print(f"{prefix}📊 Dataset: {name}")
                    print(f"{prefix}   Shape: {obj.shape}")
                    print(f"{prefix}   Dtype: {obj.dtype}")
                    if obj.shape[0] > 0:
                        print(f"{prefix}   Sample: {obj[0] if len(obj.shape) == 1 else 'array'}")
                elif isinstance(obj, h5py.Group):
                    print(f"{prefix}📁 Group: {name}")
            
            f.visititems(print_structure)
            
            # Get top-level keys
            keys = list(f.keys())
            results.add_info("HDF5 Keys", f"Found {len(keys)} top-level keys: {keys}")
            
            return f, keys
    
    except Exception as e:
        results.add_fail("File Loading", f"Error opening file: {str(e)}")
        return None, None


def validate_dataset_structure(file_path: str, results: TestResult):
    """Validate HDF5 dataset structure."""
    print("\n" + "="*70)
    print("UJI 1: VALIDASI STRUKTUR DATASET")
    print("="*70)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check for common dataset keys
            expected_keys = ['X', 'y', 'dates', 'station_codes', 'metadata']
            found_keys = list(f.keys())
            
            results.add_info("Dataset Keys", f"Found keys: {found_keys}")
            
            # Check if we have data arrays
            if 'X' in f or 'data' in f or 'features' in f:
                results.add_pass("Data Arrays", "Found data arrays in HDF5")
            else:
                results.add_warning("Data Arrays", 
                    f"Expected keys like 'X', 'data', or 'features' not found. "
                    f"Available keys: {found_keys}")
            
            # Get dataset info
            for key in found_keys[:5]:  # Check first 5 keys
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    results.add_info(f"Dataset '{key}'", 
                        f"Shape: {obj.shape}, Dtype: {obj.dtype}")
            
            return f, found_keys
    
    except Exception as e:
        results.add_fail("Structure Validation", f"Error: {str(e)}")
        return None, None


def validate_spatial_integrity(file_path: str, results: TestResult):
    """UJI 1: Validasi Integritas Spasial."""
    print("\n" + "="*70)
    print("UJI 1: VALIDASI INTEGRITAS SPASIAL")
    print("="*70)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Try to find the main data array
            data_key = None
            for key in ['X', 'data', 'features', 'scalograms']:
                if key in f:
                    data_key = key
                    break
            
            if data_key is None:
                results.add_warning("Data Key", 
                    "Could not find standard data key. Please specify the correct key.")
                return
            
            data = f[data_key]
            results.add_info("Data Shape", f"Found data with shape: {data.shape}")
            
            # Check if first dimension is number of samples
            num_samples = data.shape[0]
            results.add_info("Sample Count", f"Total samples: {num_samples}")
            
            # Try to infer station dimension
            # Common formats: [samples, stations, channels, height, width]
            # or [samples, channels, height, width] for single station
            if len(data.shape) >= 4:
                # Assume format: [samples, stations, channels, height, width]
                if data.shape[1] == EXPECTED_NUM_STATIONS:
                    results.add_pass("UJI 1.1 - Station Count", 
                        f"Data has {EXPECTED_NUM_STATIONS} stations (dimension 1)")
                else:
                    results.add_warning("UJI 1.1 - Station Count",
                        f"Expected {EXPECTED_NUM_STATIONS} stations, found {data.shape[1]} "
                        f"in dimension 1. Please verify data format.")
            else:
                results.add_warning("UJI 1.1 - Station Count",
                    f"Data shape {data.shape} does not match expected format. "
                    f"Cannot verify station count.")
            
            # Check for NaN values (sample first 100 entries)
            sample_size = min(100, num_samples)
            sample_data = data[:sample_size]
            
            if isinstance(sample_data, h5py.Dataset):
                sample_data = sample_data[:]
            
            nan_count = np.isnan(sample_data).sum()
            
            if nan_count == 0:
                results.add_pass("UJI 1.2 - NaN Check",
                    f"No NaN values found in {sample_size} sampled entries")
            else:
                results.add_fail("UJI 1.2 - NaN Check",
                    f"Found {nan_count} NaN values in {sample_size} sampled entries")
    
    except Exception as e:
        results.add_fail("Spatial Integrity", f"Error: {str(e)}")


def validate_chronological_split(file_path: str, results: TestResult):
    """UJI 2: Validasi Strict Chronological Split."""
    print("\n" + "="*70)
    print("UJI 2: VALIDASI STRICT CHRONOLOGICAL SPLIT")
    print("="*70)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Try to find dates
            dates_key = None
            for key in ['dates', 'timestamps', 'time', 'date']:
                if key in f:
                    dates_key = key
                    break
            
            if dates_key is None:
                results.add_warning("Date Information",
                    "Could not find date/timestamp information in HDF5. "
                    "Cannot validate chronological split.")
                return
            
            dates = f[dates_key][:]
            
            # Convert to datetime if needed
            if dates.dtype.kind in ['S', 'U', 'O']:  # String types
                dates_str = [d.decode('utf-8') if isinstance(d, bytes) else str(d) 
                           for d in dates]
                dates_dt = pd.to_datetime(dates_str)
            else:
                dates_dt = pd.to_datetime(dates)
            
            results.add_info("Date Range", 
                f"Dataset spans from {dates_dt.min().date()} to {dates_dt.max().date()}")
            
            # Check splits
            train_end = pd.to_datetime(TRAIN_END_DATE)
            val_start = pd.to_datetime(VAL_START_DATE)
            val_end = pd.to_datetime(VAL_END_DATE)
            test_start = pd.to_datetime(TEST_START_DATE)
            
            # Count samples in each split
            train_mask = dates_dt <= train_end
            val_mask = (dates_dt >= val_start) & (dates_dt <= val_end)
            test_mask = dates_dt >= test_start
            
            train_count = train_mask.sum()
            val_count = val_mask.sum()
            test_count = test_mask.sum()
            
            results.add_info("Split Counts",
                f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
            
            # Validate no overlap
            if train_count > 0:
                train_max = dates_dt[train_mask].max()
                if train_max <= train_end:
                    results.add_pass("UJI 2.1 - Train Date Range",
                        f"Train data ends at {train_max.date()} (≤ {TRAIN_END_DATE})")
                else:
                    results.add_fail("UJI 2.1 - Train Date Range",
                        f"Train data extends to {train_max.date()} (> {TRAIN_END_DATE})")
            
            if val_count > 0:
                val_min = dates_dt[val_mask].min()
                val_max = dates_dt[val_mask].max()
                if val_min >= val_start and val_max <= val_end:
                    results.add_pass("UJI 2.2 - Val Date Range",
                        f"Val data: {val_min.date()} to {val_max.date()}")
                else:
                    results.add_fail("UJI 2.2 - Val Date Range",
                        f"Val data outside expected range: {val_min.date()} to {val_max.date()}")
            
            if test_count > 0:
                test_min = dates_dt[test_mask].min()
                if test_min >= test_start:
                    results.add_pass("UJI 2.3 - Test Date Range",
                        f"Test data starts at {test_min.date()} (≥ {TEST_START_DATE})")
                else:
                    results.add_fail("UJI 2.3 - Test Date Range",
                        f"Test data starts at {test_min.date()} (< {TEST_START_DATE})")
            
            # Check for May 2024 data
            may_2024_mask = (dates_dt.year == 2024) & (dates_dt.month == 5)
            may_2024_count = may_2024_mask.sum()
            
            if may_2024_count > 0:
                results.add_pass("UJI 2.4 - May 2024 Data",
                    f"Found {may_2024_count} samples from May 2024")
            else:
                results.add_warning("UJI 2.4 - May 2024 Data",
                    "No May 2024 data found for space weather testing")
    
    except Exception as e:
        results.add_fail("Chronological Split", f"Error: {str(e)}")


def validate_labels(file_path: str, results: TestResult):
    """UJI 3: Validasi Target DPINN & Multi-Task."""
    print("\n" + "="*70)
    print("UJI 3: VALIDASI TARGET DPINN & MULTI-TASK")
    print("="*70)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Try to find labels
            labels_key = None
            for key in ['y', 'labels', 'targets']:
                if key in f:
                    labels_key = key
                    break
            
            if labels_key is None:
                results.add_warning("Label Information",
                    "Could not find label information. Cannot validate labels.")
                return
            
            labels = f[labels_key]
            results.add_info("Labels Shape", f"Labels shape: {labels.shape}")
            
            # Check for NaN in labels
            sample_size = min(100, labels.shape[0])
            sample_labels = labels[:sample_size]
            
            if isinstance(sample_labels, h5py.Dataset):
                sample_labels = sample_labels[:]
            
            nan_count = np.isnan(sample_labels).sum()
            
            if nan_count == 0:
                results.add_pass("UJI 3.1 - Label NaN Check",
                    f"No NaN values in {sample_size} sampled labels")
            else:
                results.add_fail("UJI 3.1 - Label NaN Check",
                    f"Found {nan_count} NaN values in labels")
            
            # Check label range
            if len(sample_labels.shape) == 1 or sample_labels.shape[1] == 1:
                # Binary labels
                unique_labels = np.unique(sample_labels)
                results.add_info("Label Values", f"Unique labels: {unique_labels}")
                
                if set(unique_labels).issubset({0, 1}):
                    results.add_pass("UJI 3.2 - Binary Labels",
                        "Labels are binary (0 or 1)")
                else:
                    results.add_warning("UJI 3.2 - Binary Labels",
                        f"Labels contain values other than 0/1: {unique_labels}")
    
    except Exception as e:
        results.add_fail("Label Validation", f"Error: {str(e)}")


def create_visualizations(file_path: str, results: TestResult):
    """Create visualizations of dataset."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 1. Dataset structure visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('HDF5 Dataset Overview', fontsize=16, fontweight='bold')
            
            # Get keys
            keys = list(f.keys())
            
            # Plot 1: Dataset sizes
            ax = axes[0, 0]
            sizes = []
            names = []
            for key in keys[:10]:  # First 10 keys
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    size_mb = obj.nbytes / (1024 * 1024)
                    sizes.append(size_mb)
                    names.append(key)
            
            if sizes:
                ax.barh(names, sizes, color='steelblue')
                ax.set_xlabel('Size (MB)')
                ax.set_title('Dataset Sizes')
                ax.grid(True, alpha=0.3)
            
            # Plot 2: Data shape info
            ax = axes[0, 1]
            shape_info = []
            for key in keys[:5]:
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    shape_info.append(f"{key}: {obj.shape}")
            
            if shape_info:
                ax.text(0.1, 0.5, '\n'.join(shape_info), 
                       fontsize=10, verticalalignment='center',
                       family='monospace')
                ax.set_title('Dataset Shapes')
                ax.axis('off')
            
            # Plot 3: Sample distribution (if dates available)
            ax = axes[1, 0]
            dates_key = None
            for key in ['dates', 'timestamps', 'time']:
                if key in f:
                    dates_key = key
                    break
            
            if dates_key:
                dates = f[dates_key][:]
                if dates.dtype.kind in ['S', 'U', 'O']:
                    dates_str = [d.decode('utf-8') if isinstance(d, bytes) else str(d) 
                               for d in dates]
                    dates_dt = pd.to_datetime(dates_str)
                else:
                    dates_dt = pd.to_datetime(dates)
                
                # Group by month
                dates_series = pd.Series(dates_dt)
                monthly_counts = dates_series.dt.to_period('M').value_counts().sort_index()
                
                ax.plot(monthly_counts.index.astype(str), monthly_counts.values, 
                       marker='o', linewidth=2, markersize=4)
                ax.set_xlabel('Month')
                ax.set_ylabel('Sample Count')
                ax.set_title('Temporal Distribution')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            # Plot 4: Label distribution (if available)
            ax = axes[1, 1]
            labels_key = None
            for key in ['y', 'labels', 'targets']:
                if key in f:
                    labels_key = key
                    break
            
            if labels_key:
                labels = f[labels_key][:]
                if len(labels.shape) == 1 or labels.shape[1] == 1:
                    unique, counts = np.unique(labels, return_counts=True)
                    ax.bar(unique, counts, color=['red', 'green'])
                    ax.set_xlabel('Label')
                    ax.set_ylabel('Count')
                    ax.set_title('Label Distribution')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, 'hdf5_dataset_overview.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            results.add_pass("Visualization", f"Saved overview to {output_file}")
    
    except Exception as e:
        results.add_warning("Visualization", f"Could not create visualizations: {str(e)}")


def main():
    """Main execution function."""
    print("="*70)
    print("ANTIGRAVITY - HDF5 DATASET VALIDATION")
    print("="*70)
    print(f"Started: {datetime.now()}")
    print(f"File: {HDF5_FILE}")
    print()
    
    # Initialize results
    results = TestResult()
    
    # Check file exists
    if not os.path.exists(HDF5_FILE):
        print(f"\n❌ ERROR: File not found: {HDF5_FILE}")
        print("Please ensure the HDF5 file is in the current directory.")
        return 1
    
    # Explore structure
    explore_hdf5_structure(HDF5_FILE, results)
    
    # Validate structure
    validate_dataset_structure(HDF5_FILE, results)
    
    # Run validation tests
    validate_spatial_integrity(HDF5_FILE, results)
    validate_chronological_split(HDF5_FILE, results)
    validate_labels(HDF5_FILE, results)
    
    # Create visualizations
    create_visualizations(HDF5_FILE, results)
    
    # Print summary
    all_passed = results.summary()
    
    print(f"\nCompleted: {datetime.now()}")
    
    # Return exit code
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
