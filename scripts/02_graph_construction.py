"""
ANTIGRAVITY Project - Script 02: Graph Construction
====================================================

Membuat time-based graph snapshots untuk setiap hari observasi.
Setiap snapshot berisi 24 nodes (stasiun) dengan edge connections
berdasarkan jarak geografis dan penalti tektonik.

Author: ANTIGRAVITY Team
Date: 2026-05-02
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import math
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import json

# Configuration
PROCESSED_DATA_DIR = 'data/processed'
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'dataset_graphs.pt')
LOG_FILE = os.path.join(PROCESSED_DATA_DIR, 'graph_construction_log.txt')

# Prekursor window (days before earthquake)
PRECURSOR_WINDOW_DAYS = 14

# Minimum magnitude for significant earthquakes
MIN_SIGNIFICANT_MAG = 5.0

# Station clustering (tectonic regions)
STATION_CLUSTERS = {
    'Sunda': ['SBG', 'MLB', 'SCN', 'KPY', 'LWA', 'LPS', 'SRG', 'SKB', 'CLP', 'YOG', 'TRT', 'GSI'],
    'Wallacea': ['TND', 'GTO', 'LWK', 'PLU', 'TNT', 'TRD', 'LUT', 'ALR', 'ROT'],
    'Sahul': ['SMI', 'AMB', 'JYP', 'SRO']
}

# Earth radius in km
EARTH_RADIUS_KM = 6371.0


class GraphConstructionLogger:
    """Logger untuk graph construction process."""
    
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write(f"ANTIGRAVITY Graph Construction Log\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write("="*60 + "\n\n")
    
    def log(self, message):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        Distance in kilometers
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = EARTH_RADIUS_KM * c
    return distance


def calculate_azimuth(lat1, lon1, lat2, lon2):
    """
    Calculate azimuth (bearing) from point 1 to point 2.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        Azimuth in degrees (0-360)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
    
    azimuth_rad = math.atan2(x, y)
    azimuth_deg = math.degrees(azimuth_rad)
    
    # Normalize to 0-360
    azimuth_deg = (azimuth_deg + 360) % 360
    
    return azimuth_deg


def assign_region_index(station_code):
    """
    Assign tectonic region index to station.
    
    Args:
        station_code: Station code (e.g., 'SBG')
    
    Returns:
        region_idx: 0=Sunda, 1=Wallacea, 2=Sahul
    """
    for idx, (region, stations) in enumerate(STATION_CLUSTERS.items()):
        if station_code in stations:
            return idx
    
    # Default to Sunda if not found
    return 0


def calculate_tectonic_penalty(region_i, region_j):
    """
    Calculate penalty for edges crossing tectonic boundaries.
    
    Args:
        region_i: Region index of station i
        region_j: Region index of station j
    
    Returns:
        P_fault: 0.5 if different regions, 0.0 if same
    """
    if region_i != region_j:
        return 0.5  # Penalty for cross-tectonic edge
    else:
        return 0.0  # No penalty for same-region edge


def build_adjacency_matrix(stations_df, logger):
    """
    Build adjacency matrix and edge attributes for station graph.
    
    Args:
        stations_df: DataFrame with station information
        logger: Logger instance
    
    Returns:
        edge_index: [2, num_edges] tensor
        edge_attr: [num_edges, 3] tensor (distance, penalty, weight)
        station_info: dict with station metadata
    """
    logger.log("Building adjacency matrix...")
    
    num_stations = len(stations_df)
    
    # Assign region indices
    stations_df['region_idx'] = stations_df['Kode Stasiun'].apply(assign_region_index)
    
    # Build fully connected graph
    edge_list = []
    edge_attributes = []
    
    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            station_i = stations_df.iloc[i]
            station_j = stations_df.iloc[j]
            
            # Calculate distance
            distance = haversine_distance(
                station_i['Latitude'], station_i['Longitude'],
                station_j['Latitude'], station_j['Longitude']
            )
            
            # Calculate tectonic penalty
            penalty = calculate_tectonic_penalty(
                station_i['region_idx'],
                station_j['region_idx']
            )
            
            # Calculate edge weight (inverse distance)
            weight = 1.0 / (distance + 1e-6)
            
            # Add edge (undirected: both directions)
            edge_list.append([i, j])
            edge_list.append([j, i])
            
            edge_attributes.append([distance, penalty, weight])
            edge_attributes.append([distance, penalty, weight])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    
    logger.log(f"✓ Adjacency matrix built: {num_stations} nodes, {len(edge_list)} edges")
    
    # Create station info dict
    station_info = {
        'codes': stations_df['Kode Stasiun'].tolist(),
        'latitudes': stations_df['Latitude'].tolist(),
        'longitudes': stations_df['Longitude'].tolist(),
        'elevations': stations_df.get('Elevation', [0]*num_stations).tolist(),
        'region_indices': stations_df['region_idx'].tolist()
    }
    
    return edge_index, edge_attr, station_info


def extract_labels(date_t, earthquake_df, station_info, precursor_days=PRECURSOR_WINDOW_DAYS):
    """
    Extract labels for graph snapshot on date t.
    
    Args:
        date_t: Date of snapshot
        earthquake_df: DataFrame with earthquake catalog
        station_info: Dict with station metadata
        precursor_days: Precursor window in days
    
    Returns:
        labels: dict with event, magnitude, azimuth, distances
    """
    # Find earthquakes in future window [t, t+precursor_days]
    future_window_start = date_t
    future_window_end = date_t + timedelta(days=precursor_days)
    
    future_events = earthquake_df[
        (earthquake_df['time'] >= future_window_start) &
        (earthquake_df['time'] <= future_window_end) &
        (earthquake_df['mag'] >= MIN_SIGNIFICANT_MAG)
    ]
    
    if len(future_events) == 0:
        # Background noise
        return {
            'event': 0,
            'magnitude': 0.0,
            'azimuth': 0.0,
            'distances': [0.0] * len(station_info['codes'])
        }
    
    # Take earliest earthquake in window
    earthquake = future_events.iloc[0]
    
    # Calculate centroid of active stations
    centroid_lat = np.mean(station_info['latitudes'])
    centroid_lon = np.mean(station_info['longitudes'])
    
    # Calculate azimuth from centroid to epicenter
    azimuth = calculate_azimuth(
        centroid_lat, centroid_lon,
        earthquake['latitude'], earthquake['longitude']
    )
    
    # Calculate distance from each station to epicenter
    distances = []
    for lat, lon in zip(station_info['latitudes'], station_info['longitudes']):
        dist = haversine_distance(
            lat, lon,
            earthquake['latitude'], earthquake['longitude']
        )
        distances.append(dist)
    
    return {
        'event': 1,
        'magnitude': float(earthquake['mag']),
        'azimuth': azimuth,
        'distances': distances
    }


def create_graph_snapshot(date_t, edge_index, edge_attr, station_info, 
                         earthquake_df, logger=None):
    """
    Create graph snapshot for a specific date.
    
    Args:
        date_t: Date of snapshot
        edge_index: Edge connectivity
        edge_attr: Edge attributes
        station_info: Station metadata
        earthquake_df: Earthquake catalog
        logger: Logger instance (optional)
    
    Returns:
        Data: PyTorch Geometric Data object
    """
    num_stations = len(station_info['codes'])
    
    # Create node features (basic: lat, lon, elevation, region_idx, is_active)
    # In real implementation, add seismic features here
    node_features = []
    for i in range(num_stations):
        features = [
            station_info['latitudes'][i],
            station_info['longitudes'][i],
            station_info['elevations'][i],
            station_info['region_indices'][i],
            1.0  # is_active (assume all active for now)
        ]
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Extract labels
    labels = extract_labels(date_t, earthquake_df, station_info)
    
    # Create graph data object
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_event=torch.tensor([labels['event']], dtype=torch.float),
        y_mag=torch.tensor([labels['magnitude']], dtype=torch.float),
        y_azm=torch.tensor([labels['azimuth']], dtype=torch.float),
        y_dist=torch.tensor(labels['distances'], dtype=torch.float),
        date=date_t.strftime('%Y-%m-%d'),
        # Placeholder for space weather (to be added later)
        kp_index=0.0,
        dst_index=0.0
    )
    
    return graph


def generate_date_range(start_date, end_date):
    """
    Generate list of dates from start to end.
    
    Args:
        start_date: Start date (datetime or string)
        end_date: End date (datetime or string)
    
    Returns:
        List of datetime objects
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return dates.tolist()


def main():
    """Main execution function."""
    print("="*60)
    print("ANTIGRAVITY - Graph Construction Script")
    print("="*60)
    print()
    
    # Initialize logger
    logger = GraphConstructionLogger(LOG_FILE)
    
    try:
        # Load cleaned data
        logger.log("Loading cleaned data...")
        
        stations_file = os.path.join(PROCESSED_DATA_DIR, 'lokasi_stasiun_clean.csv')
        earthquake_file = os.path.join(PROCESSED_DATA_DIR, 'earthquake_catalog_clean.csv')
        
        if not os.path.exists(stations_file):
            logger.log(f"ERROR: File not found: {stations_file}")
            logger.log("Please run 01_data_cleaning.py first")
            return 1
        
        if not os.path.exists(earthquake_file):
            logger.log(f"ERROR: File not found: {earthquake_file}")
            logger.log("Please run 01_data_cleaning.py first")
            return 1
        
        stations_df = pd.read_csv(stations_file)
        earthquake_df = pd.read_csv(earthquake_file)
        earthquake_df['time'] = pd.to_datetime(earthquake_df['time'])
        
        logger.log(f"✓ Loaded {len(stations_df)} stations")
        logger.log(f"✓ Loaded {len(earthquake_df)} earthquake events")
        
        # Build adjacency matrix
        edge_index, edge_attr, station_info = build_adjacency_matrix(stations_df, logger)
        
        # Determine date range
        start_date = earthquake_df['time'].min().date()
        end_date = earthquake_df['time'].max().date()
        
        logger.log(f"\nGenerating graph snapshots from {start_date} to {end_date}")
        
        dates = generate_date_range(start_date, end_date)
        logger.log(f"Total days to process: {len(dates)}")
        
        # Create graph snapshots
        logger.log("\nCreating graph snapshots...")
        graphs = []
        
        for date_t in tqdm(dates, desc="Processing days"):
            graph = create_graph_snapshot(
                date_t, edge_index, edge_attr, station_info, earthquake_df
            )
            graphs.append(graph)
        
        logger.log(f"✓ Created {len(graphs)} graph snapshots")
        
        # Statistics
        event_count = sum([1 for g in graphs if g.y_event.item() == 1])
        background_count = len(graphs) - event_count
        
        logger.log(f"\nDataset Statistics:")
        logger.log(f"  Total snapshots: {len(graphs)}")
        logger.log(f"  Event (prekursor): {event_count} ({event_count/len(graphs)*100:.2f}%)")
        logger.log(f"  Background: {background_count} ({background_count/len(graphs)*100:.2f}%)")
        logger.log(f"  Imbalance ratio: 1:{background_count/max(event_count, 1):.1f}")
        
        # Magnitude distribution
        magnitudes = [g.y_mag.item() for g in graphs if g.y_event.item() == 1]
        if magnitudes:
            logger.log(f"\nMagnitude Distribution (events only):")
            logger.log(f"  Count: {len(magnitudes)}")
            logger.log(f"  Mean: {np.mean(magnitudes):.2f}")
            logger.log(f"  Std: {np.std(magnitudes):.2f}")
            logger.log(f"  Min: {np.min(magnitudes):.2f}")
            logger.log(f"  Max: {np.max(magnitudes):.2f}")
        
        # Save graphs
        logger.log(f"\nSaving graph dataset to {OUTPUT_FILE}...")
        torch.save(graphs, OUTPUT_FILE)
        logger.log(f"✓ Saved {len(graphs)} graphs")
        
        # Save metadata
        metadata = {
            'num_graphs': len(graphs),
            'num_stations': len(station_info['codes']),
            'num_edges': edge_index.shape[1],
            'date_range': [str(start_date), str(end_date)],
            'event_count': event_count,
            'background_count': background_count,
            'precursor_window_days': PRECURSOR_WINDOW_DAYS,
            'min_significant_mag': MIN_SIGNIFICANT_MAG,
            'station_clusters': STATION_CLUSTERS,
            'created': datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(PROCESSED_DATA_DIR, 'dataset_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.log(f"✓ Saved metadata to {metadata_file}")
        
        logger.log("\n" + "="*60)
        logger.log("Graph construction completed successfully!")
        logger.log("="*60)
        logger.log("\nNext Steps:")
        logger.log("1. Review graph_construction_log.txt for details")
        logger.log("2. Run script 03_chronological_split.py")
        
        return 0
    
    except Exception as e:
        logger.log(f"\nERROR: Unexpected error occurred: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
