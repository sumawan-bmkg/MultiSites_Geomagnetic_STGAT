"""
ANTIGRAVITY Project - Script 03: Chronological Split
=====================================================

Membagi dataset graph snapshots ke dalam train/val/test sets
secara ketat berdasarkan waktu global untuk menghindari data leakage.

Split:
- Train: hingga 31 Des 2023
- Val: 1 Jan 2024 - 31 Mar 2025
- Test: mulai 1 Jan 2026

Author: ANTIGRAVITY Team
Date: 2026-05-02
"""

import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

# Configuration
PROCESSED_DATA_DIR = 'data/processed'
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'

INPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'dataset_graphs.pt')
LOG_FILE = os.path.join(PROCESSED_DATA_DIR, 'chronological_split_log.txt')

# Split dates
TRAIN_END = '2023-12-31'
VAL_END = '2025-03-31'
TEST_START = '2026-01-01'


class SplitLogger:
    """Logger untuk chronological split process."""
    
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write(f"ANTIGRAVITY Chronological Split Log\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write("="*60 + "\n\n")
    
    def log(self, message):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")


def load_graphs(input_file, logger):
    """
    Load all graph snapshots from file.
    
    Args:
        input_file: Path to graphs file
        logger: Logger instance
    
    Returns:
        List of Data objects
    """
    logger.log(f"Loading graphs from {input_file}...")
    
    if not os.path.exists(input_file):
        logger.log(f"ERROR: File not found: {input_file}")
        logger.log("Please run 02_graph_construction.py first")
        return None
    
    graphs = torch.load(input_file)
    logger.log(f"✓ Loaded {len(graphs)} graph snapshots")
    
    return graphs


def chronological_split(graphs, train_end, val_end, test_start, logger):
    """
    Split graphs based on global date.
    
    Args:
        graphs: List of Data objects with 'date' attribute
        train_end: Last date for training set (string)
        val_end: Last date for validation set (string)
        test_start: First date for test set (string)
        logger: Logger instance
    
    Returns:
        train_graphs, val_graphs, test_graphs
    """
    logger.log("\nPerforming chronological split...")
    logger.log(f"  Train: up to {train_end}")
    logger.log(f"  Val: {train_end} to {val_end}")
    logger.log(f"  Test: from {test_start}")
    
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    test_start_dt = pd.to_datetime(test_start)
    
    train_graphs = []
    val_graphs = []
    test_graphs = []
    
    for graph in graphs:
        graph_date = pd.to_datetime(graph.date)
        
        if graph_date <= train_end_dt:
            train_graphs.append(graph)
        elif graph_date <= val_end_dt:
            val_graphs.append(graph)
        elif graph_date >= test_start_dt:
            test_graphs.append(graph)
        # Dates between val_end and test_start are discarded (gap period)
    
    logger.log(f"\n✓ Split completed:")
    logger.log(f"  Train: {len(train_graphs)} graphs")
    logger.log(f"  Val: {len(val_graphs)} graphs")
    logger.log(f"  Test: {len(test_graphs)} graphs")
    
    return train_graphs, val_graphs, test_graphs


def validate_split(train_graphs, val_graphs, test_graphs, logger):
    """
    Validate that split is truly chronological with no overlap.
    
    Args:
        train_graphs, val_graphs, test_graphs: Split datasets
        logger: Logger instance
    
    Returns:
        bool: True if validation passed
    """
    logger.log("\nValidating split...")
    
    # Extract dates
    train_dates = [pd.to_datetime(g.date) for g in train_graphs]
    val_dates = [pd.to_datetime(g.date) for g in val_graphs]
    test_dates = [pd.to_datetime(g.date) for g in test_graphs]
    
    # Check 1: No overlap
    if train_dates and val_dates:
        train_max = max(train_dates)
        val_min = min(val_dates)
        
        if train_max >= val_min:
            logger.log(f"✗ ERROR: Train and Val overlap!")
            logger.log(f"  Train max: {train_max}")
            logger.log(f"  Val min: {val_min}")
            return False
    
    if val_dates and test_dates:
        val_max = max(val_dates)
        test_min = min(test_dates)
        
        if val_max >= test_min:
            logger.log(f"✗ ERROR: Val and Test overlap!")
            logger.log(f"  Val max: {val_max}")
            logger.log(f"  Test min: {test_min}")
            return False
    
    logger.log("✓ No temporal overlap between splits")
    
    # Check 2: Chronological order within each split
    if train_dates != sorted(train_dates):
        logger.log("✗ ERROR: Train set not chronologically ordered!")
        return False
    
    if val_dates != sorted(val_dates):
        logger.log("✗ ERROR: Val set not chronologically ordered!")
        return False
    
    if test_dates != sorted(test_dates):
        logger.log("✗ ERROR: Test set not chronologically ordered!")
        return False
    
    logger.log("✓ Each split is chronologically ordered")
    
    # Check 3: Print date ranges
    logger.log("\nDate Ranges:")
    if train_dates:
        logger.log(f"  Train: {min(train_dates).date()} to {max(train_dates).date()}")
    if val_dates:
        logger.log(f"  Val:   {min(val_dates).date()} to {max(val_dates).date()}")
    if test_dates:
        logger.log(f"  Test:  {min(test_dates).date()} to {max(test_dates).date()}")
    
    return True


def analyze_class_distribution(graphs, split_name, logger):
    """
    Analyze event vs background distribution.
    
    Args:
        graphs: List of Data objects
        split_name: Name of split (e.g., 'Train')
        logger: Logger instance
    
    Returns:
        dict: Statistics
    """
    if not graphs:
        logger.log(f"\n{split_name} Set: No graphs")
        return {}
    
    event_count = sum([1 for g in graphs if g.y_event.item() == 1])
    background_count = len(graphs) - event_count
    event_ratio = event_count / len(graphs) if len(graphs) > 0 else 0
    
    logger.log(f"\n{split_name} Set Class Distribution:")
    logger.log(f"  Total graphs: {len(graphs)}")
    logger.log(f"  Event (prekursor): {event_count} ({event_ratio*100:.2f}%)")
    logger.log(f"  Background: {background_count} ({(1-event_ratio)*100:.2f}%)")
    
    if event_count > 0:
        imbalance_ratio = background_count / event_count
        logger.log(f"  Imbalance ratio: 1:{imbalance_ratio:.1f}")
    
    return {
        'total': len(graphs),
        'event': event_count,
        'background': background_count,
        'event_ratio': event_ratio
    }


def analyze_magnitude_distribution(graphs, split_name, logger):
    """
    Analyze magnitude distribution for events.
    
    Args:
        graphs: List of Data objects
        split_name: Name of split
        logger: Logger instance
    
    Returns:
        dict: Statistics
    """
    magnitudes = [g.y_mag.item() for g in graphs if g.y_event.item() == 1]
    
    if len(magnitudes) == 0:
        logger.log(f"\n{split_name} Set Magnitude: No events")
        return {}
    
    logger.log(f"\n{split_name} Set Magnitude Distribution:")
    logger.log(f"  Count: {len(magnitudes)}")
    logger.log(f"  Mean: {np.mean(magnitudes):.2f}")
    logger.log(f"  Std: {np.std(magnitudes):.2f}")
    logger.log(f"  Min: {np.min(magnitudes):.2f}")
    logger.log(f"  Max: {np.max(magnitudes):.2f}")
    logger.log(f"  Median: {np.median(magnitudes):.2f}")
    
    # Histogram
    bins = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 10.0]
    hist, _ = np.histogram(magnitudes, bins=bins)
    
    logger.log("\n  Magnitude Bins:")
    for i in range(len(bins)-1):
        logger.log(f"    {bins[i]:.1f} - {bins[i+1]:.1f}: {hist[i]} events")
    
    return {
        'count': len(magnitudes),
        'mean': float(np.mean(magnitudes)),
        'std': float(np.std(magnitudes)),
        'min': float(np.min(magnitudes)),
        'max': float(np.max(magnitudes)),
        'median': float(np.median(magnitudes))
    }


def analyze_space_weather(val_graphs, logger):
    """
    Analyze space weather events in validation set.
    
    Args:
        val_graphs: Validation set graphs
        logger: Logger instance
    """
    if not val_graphs:
        return
    
    logger.log("\nSpace Weather Analysis (Validation Set):")
    
    # Extract Kp values (if available)
    kp_values = [g.kp_index for g in val_graphs if hasattr(g, 'kp_index')]
    
    if not kp_values:
        logger.log("  No space weather data available")
        return
    
    # Find extreme events (Kp >= 7)
    dates = [pd.to_datetime(g.date) for g in val_graphs]
    extreme_kp = [(d, kp) for d, kp in zip(dates, kp_values) if kp >= 7.0]
    
    logger.log(f"  Total days: {len(val_graphs)}")
    logger.log(f"  Days with Kp ≥ 7: {len(extreme_kp)}")
    
    if extreme_kp:
        logger.log("\n  Extreme Space Weather Events:")
        for date, kp in extreme_kp[:10]:  # Show first 10
            logger.log(f"    {date.date()}: Kp = {kp}")
    
    # Check for May 2024 storm
    may_2024 = [(d, kp) for d, kp in zip(dates, kp_values) 
                if d.year == 2024 and d.month == 5]
    
    if may_2024:
        max_kp_may = max([kp for _, kp in may_2024])
        logger.log(f"\n  May 2024 Maximum Kp: {max_kp_may}")
        if max_kp_may >= 9.0:
            logger.log("  ✓ Extreme storm (Kp≥9.0) detected in validation set")


def save_split_datasets(train_graphs, val_graphs, test_graphs, logger):
    """
    Save split datasets to separate directories.
    
    Args:
        train_graphs, val_graphs, test_graphs: Split datasets
        logger: Logger instance
    """
    logger.log("\nSaving split datasets...")
    
    # Create directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Save graphs
    torch.save(train_graphs, os.path.join(TRAIN_DIR, 'graphs.pt'))
    torch.save(val_graphs, os.path.join(VAL_DIR, 'graphs.pt'))
    torch.save(test_graphs, os.path.join(TEST_DIR, 'graphs.pt'))
    
    logger.log(f"✓ Saved train set: {len(train_graphs)} graphs")
    logger.log(f"✓ Saved val set: {len(val_graphs)} graphs")
    logger.log(f"✓ Saved test set: {len(test_graphs)} graphs")
    
    # Save metadata
    metadata = {
        'train': {
            'count': len(train_graphs),
            'date_range': [
                str(min([pd.to_datetime(g.date) for g in train_graphs]).date()) if train_graphs else None,
                str(max([pd.to_datetime(g.date) for g in train_graphs]).date()) if train_graphs else None
            ],
            'file': os.path.join(TRAIN_DIR, 'graphs.pt')
        },
        'val': {
            'count': len(val_graphs),
            'date_range': [
                str(min([pd.to_datetime(g.date) for g in val_graphs]).date()) if val_graphs else None,
                str(max([pd.to_datetime(g.date) for g in val_graphs]).date()) if val_graphs else None
            ],
            'file': os.path.join(VAL_DIR, 'graphs.pt')
        },
        'test': {
            'count': len(test_graphs),
            'date_range': [
                str(min([pd.to_datetime(g.date) for g in test_graphs]).date()) if test_graphs else None,
                str(max([pd.to_datetime(g.date) for g in test_graphs]).date()) if test_graphs else None
            ],
            'file': os.path.join(TEST_DIR, 'graphs.pt')
        },
        'split_dates': {
            'train_end': TRAIN_END,
            'val_end': VAL_END,
            'test_start': TEST_START
        },
        'created': datetime.now().isoformat()
    }
    
    metadata_file = os.path.join(PROCESSED_DATA_DIR, 'split_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.log(f"✓ Saved split metadata to {metadata_file}")


def main():
    """Main execution function."""
    print("="*60)
    print("ANTIGRAVITY - Chronological Split Script")
    print("="*60)
    print()
    
    # Initialize logger
    logger = SplitLogger(LOG_FILE)
    
    try:
        # Step 1: Load graphs
        logger.log("[1/5] Loading graph snapshots...")
        graphs = load_graphs(INPUT_FILE, logger)
        
        if graphs is None:
            return 1
        
        # Step 2: Chronological split
        logger.log("\n[2/5] Performing chronological split...")
        train_graphs, val_graphs, test_graphs = chronological_split(
            graphs, TRAIN_END, VAL_END, TEST_START, logger
        )
        
        # Step 3: Validate split
        logger.log("\n[3/5] Validating split...")
        if not validate_split(train_graphs, val_graphs, test_graphs, logger):
            logger.log("\n✗ Validation failed!")
            return 1
        
        # Step 4: Analyze distributions
        logger.log("\n[4/5] Analyzing distributions...")
        
        train_class_stats = analyze_class_distribution(train_graphs, 'Train', logger)
        val_class_stats = analyze_class_distribution(val_graphs, 'Validation', logger)
        test_class_stats = analyze_class_distribution(test_graphs, 'Test', logger)
        
        train_mag_stats = analyze_magnitude_distribution(train_graphs, 'Train', logger)
        val_mag_stats = analyze_magnitude_distribution(val_graphs, 'Validation', logger)
        test_mag_stats = analyze_magnitude_distribution(test_graphs, 'Test', logger)
        
        analyze_space_weather(val_graphs, logger)
        
        # Step 5: Save split datasets
        logger.log("\n[5/5] Saving split datasets...")
        save_split_datasets(train_graphs, val_graphs, test_graphs, logger)
        
        logger.log("\n" + "="*60)
        logger.log("Chronological split completed successfully!")
        logger.log("="*60)
        logger.log("\nNext Steps:")
        logger.log("1. Review chronological_split_log.txt for details")
        logger.log("2. Implement model architecture")
        logger.log("3. Run training script")
        
        return 0
    
    except Exception as e:
        logger.log(f"\nERROR: Unexpected error occurred: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
