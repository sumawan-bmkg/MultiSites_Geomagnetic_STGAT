"""
ANTIGRAVITY Project - Script 01: Data Cleaning
===============================================

Membersihkan dan memvalidasi data mentah:
1. lokasi_stasiun.csv - Koordinat 24 stasiun seismik
2. earthquake_catalog_2018_2025_merged_robust.csv - Katalog gempa

Author: ANTIGRAVITY Team
Date: 2026-05-02
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys

# Configuration
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
LOG_FILE = os.path.join(PROCESSED_DATA_DIR, 'cleaning_log.txt')

# Expected station count
EXPECTED_STATION_COUNT = 24

# Indonesia coordinate bounds
LAT_MIN, LAT_MAX = -11.0, 6.0
LON_MIN, LON_MAX = 95.0, 141.0

# Earthquake coordinate bounds (wider for regional events)
EQ_LAT_MIN, EQ_LAT_MAX = -15.0, 10.0
EQ_LON_MIN, EQ_LON_MAX = 90.0, 145.0


class CleaningLogger:
    """Logger untuk mencatat semua operasi cleaning."""
    
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write(f"ANTIGRAVITY Data Cleaning Log\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write("="*60 + "\n\n")
    
    def log(self, message):
        """Log message ke file dan print ke console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")


def clean_station_data(logger):
    """
    Membersihkan lokasi_stasiun.csv.
    
    Returns:
        DataFrame: Cleaned station data
    """
    logger.log("="*60)
    logger.log("CLEANING STATION DATA")
    logger.log("="*60)
    
    # Load data
    station_file = os.path.join(RAW_DATA_DIR, 'lokasi_stasiun.csv')
    
    if not os.path.exists(station_file):
        logger.log(f"ERROR: File not found: {station_file}")
        logger.log("Please place lokasi_stasiun.csv in data/raw/ directory")
        return None
    
    logger.log(f"Loading: {station_file}")
    
    # Try multiple encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(station_file, encoding=encoding)
            logger.log(f"Successfully loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        logger.log("ERROR: Could not read file with any encoding")
        return None
    
    logger.log(f"Initial row count: {len(df)}")
    logger.log(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['Kode Stasiun', 'Latitude', 'Longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.log(f"ERROR: Missing required columns: {missing_cols}")
        return None
    
    # Remove rows with NaN in critical columns
    initial_count = len(df)
    df = df.dropna(subset=['Kode Stasiun', 'Latitude', 'Longitude'])
    removed_nan = initial_count - len(df)
    
    if removed_nan > 0:
        logger.log(f"Removed {removed_nan} rows with NaN in critical columns")
    
    # Fill Elevation NaN with 0
    if 'Elevation' in df.columns:
        elevation_nan_count = df['Elevation'].isna().sum()
        if elevation_nan_count > 0:
            logger.log(f"Filling {elevation_nan_count} NaN Elevation values with 0")
            df['Elevation'] = df['Elevation'].fillna(0)
    
    # Validate coordinates
    invalid_coords = df[
        (df['Latitude'] < LAT_MIN) | (df['Latitude'] > LAT_MAX) |
        (df['Longitude'] < LON_MIN) | (df['Longitude'] > LON_MAX)
    ]
    
    if len(invalid_coords) > 0:
        logger.log(f"WARNING: Found {len(invalid_coords)} stations with invalid coordinates:")
        for idx, row in invalid_coords.iterrows():
            logger.log(f"  {row['Kode Stasiun']}: ({row['Latitude']}, {row['Longitude']})")
        
        df = df[
            (df['Latitude'] >= LAT_MIN) & (df['Latitude'] <= LAT_MAX) &
            (df['Longitude'] >= LON_MIN) & (df['Longitude'] <= LON_MAX)
        ]
        logger.log(f"Removed {len(invalid_coords)} stations with invalid coordinates")
    
    # Check for duplicates
    duplicates = df[df.duplicated(subset=['Kode Stasiun'], keep=False)]
    if len(duplicates) > 0:
        logger.log(f"WARNING: Found {len(duplicates)} duplicate station codes:")
        logger.log(f"{duplicates['Kode Stasiun'].tolist()}")
        df = df.drop_duplicates(subset=['Kode Stasiun'], keep='first')
        logger.log(f"Kept first occurrence of each duplicate")
    
    # Validate station count
    final_count = len(df)
    logger.log(f"Final station count: {final_count}")
    
    if final_count != EXPECTED_STATION_COUNT:
        logger.log(f"WARNING: Expected {EXPECTED_STATION_COUNT} stations, got {final_count}")
    else:
        logger.log(f"✓ Station count matches expected: {EXPECTED_STATION_COUNT}")
    
    # Save cleaned data
    output_file = os.path.join(PROCESSED_DATA_DIR, 'lokasi_stasiun_clean.csv')
    df.to_csv(output_file, index=False)
    logger.log(f"✓ Saved cleaned station data: {output_file}")
    
    # Print station list
    logger.log("\nCleaned Station List:")
    for idx, row in df.iterrows():
        logger.log(f"  {row['Kode Stasiun']}: ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
    
    return df


def clean_earthquake_catalog(logger):
    """
    Membersihkan earthquake_catalog_2018_2025_merged_robust.csv.
    
    Returns:
        DataFrame: Cleaned earthquake catalog
    """
    logger.log("\n" + "="*60)
    logger.log("CLEANING EARTHQUAKE CATALOG")
    logger.log("="*60)
    
    # Load data
    catalog_file = os.path.join(RAW_DATA_DIR, 'earthquake_catalog_2018_2025_merged_robust.csv')
    
    if not os.path.exists(catalog_file):
        logger.log(f"ERROR: File not found: {catalog_file}")
        logger.log("Please place earthquake_catalog_2018_2025_merged_robust.csv in data/raw/ directory")
        return None
    
    logger.log(f"Loading: {catalog_file}")
    
    # Try multiple encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(catalog_file, encoding=encoding)
            logger.log(f"Successfully loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        logger.log("ERROR: Could not read file with any encoding")
        return None
    
    logger.log(f"Initial row count: {len(df)}")
    logger.log(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['time', 'latitude', 'longitude', 'depth', 'mag']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.log(f"ERROR: Missing required columns: {missing_cols}")
        return None
    
    # Convert time to datetime
    logger.log("Converting time column to datetime...")
    try:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        logger.log(f"✓ Time conversion successful")
        logger.log(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    except Exception as e:
        logger.log(f"ERROR: Time conversion failed: {e}")
        return None
    
    # Remove training artifacts (station column with numeric strings)
    if 'station' in df.columns:
        initial_count = len(df)
        # Remove rows where station is purely numeric (e.g., '20240125')
        df = df[~df['station'].astype(str).str.match(r'^\d+$', na=False)]
        removed_artifacts = initial_count - len(df)
        
        if removed_artifacts > 0:
            logger.log(f"Removed {removed_artifacts} rows with training artifact (numeric station codes)")
    
    # Remove rows with NaN in critical columns
    initial_count = len(df)
    df = df.dropna(subset=['time', 'latitude', 'longitude', 'depth', 'mag'])
    removed_nan = initial_count - len(df)
    
    if removed_nan > 0:
        logger.log(f"Removed {removed_nan} rows with NaN in critical columns")
    
    # Validate coordinates
    invalid_coords = df[
        (df['latitude'] < EQ_LAT_MIN) | (df['latitude'] > EQ_LAT_MAX) |
        (df['longitude'] < EQ_LON_MIN) | (df['longitude'] > EQ_LON_MAX)
    ]
    
    if len(invalid_coords) > 0:
        logger.log(f"Removed {len(invalid_coords)} events with coordinates outside region")
        df = df[
            (df['latitude'] >= EQ_LAT_MIN) & (df['latitude'] <= EQ_LAT_MAX) &
            (df['longitude'] >= EQ_LON_MIN) & (df['longitude'] <= EQ_LON_MAX)
        ]
    
    # Validate depth (should be positive)
    invalid_depth = df[df['depth'] < 0]
    if len(invalid_depth) > 0:
        logger.log(f"WARNING: Found {len(invalid_depth)} events with negative depth")
        df = df[df['depth'] >= 0]
        logger.log(f"Removed events with negative depth")
    
    # Validate magnitude (reasonable range)
    invalid_mag = df[(df['mag'] < 0) | (df['mag'] > 10)]
    if len(invalid_mag) > 0:
        logger.log(f"WARNING: Found {len(invalid_mag)} events with unrealistic magnitude")
        df = df[(df['mag'] >= 0) & (df['mag'] <= 10)]
        logger.log(f"Removed events with unrealistic magnitude")
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Statistics
    logger.log(f"\nFinal earthquake count: {len(df)}")
    logger.log(f"Date range: {df['time'].min()} to {df['time'].max()}")
    logger.log(f"Magnitude range: {df['mag'].min():.2f} to {df['mag'].max():.2f}")
    logger.log(f"Depth range: {df['depth'].min():.2f} to {df['depth'].max():.2f} km")
    
    # Magnitude distribution
    logger.log("\nMagnitude Distribution:")
    mag_bins = [0, 3, 4, 5, 6, 7, 8, 10]
    for i in range(len(mag_bins)-1):
        count = len(df[(df['mag'] >= mag_bins[i]) & (df['mag'] < mag_bins[i+1])])
        logger.log(f"  {mag_bins[i]:.1f} - {mag_bins[i+1]:.1f}: {count} events")
    
    # Save cleaned data
    output_file = os.path.join(PROCESSED_DATA_DIR, 'earthquake_catalog_clean.csv')
    df.to_csv(output_file, index=False)
    logger.log(f"\n✓ Saved cleaned earthquake catalog: {output_file}")
    
    # Save significant events (Mw >= 5.0)
    significant = df[df['mag'] >= 5.0]
    significant_file = os.path.join(PROCESSED_DATA_DIR, 'earthquake_catalog_significant.csv')
    significant.to_csv(significant_file, index=False)
    logger.log(f"✓ Saved significant events (Mw >= 5.0): {significant_file}")
    logger.log(f"  Count: {len(significant)} events")
    
    return df


def generate_summary_report(station_df, earthquake_df, logger):
    """Generate summary report of cleaning process."""
    logger.log("\n" + "="*60)
    logger.log("CLEANING SUMMARY")
    logger.log("="*60)
    
    if station_df is not None:
        logger.log(f"✓ Station data cleaned: {len(station_df)} stations")
    else:
        logger.log("✗ Station data cleaning failed")
    
    if earthquake_df is not None:
        logger.log(f"✓ Earthquake catalog cleaned: {len(earthquake_df)} events")
        significant_count = len(earthquake_df[earthquake_df['mag'] >= 5.0])
        logger.log(f"  Significant events (Mw >= 5.0): {significant_count}")
    else:
        logger.log("✗ Earthquake catalog cleaning failed")
    
    logger.log("\nNext Steps:")
    logger.log("1. Review cleaning_log.txt for details")
    logger.log("2. Verify cleaned data in data/processed/")
    logger.log("3. Run script 02_graph_construction.py")
    
    logger.log(f"\nCompleted: {datetime.now()}")


def main():
    """Main execution function."""
    print("="*60)
    print("ANTIGRAVITY - Data Cleaning Script")
    print("="*60)
    print()
    
    # Create output directory
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Initialize logger
    logger = CleaningLogger(LOG_FILE)
    
    try:
        # Clean station data
        station_df = clean_station_data(logger)
        
        # Clean earthquake catalog
        earthquake_df = clean_earthquake_catalog(logger)
        
        # Generate summary
        generate_summary_report(station_df, earthquake_df, logger)
        
        # Check if both succeeded
        if station_df is not None and earthquake_df is not None:
            logger.log("\n✓ Data cleaning completed successfully!")
            return 0
        else:
            logger.log("\n✗ Data cleaning completed with errors")
            return 1
    
    except Exception as e:
        logger.log(f"\nERROR: Unexpected error occurred: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
