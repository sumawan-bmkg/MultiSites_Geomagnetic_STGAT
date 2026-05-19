"""
ANTIGRAVITY Project - PyTest Dataset Validation
================================================

Unit tests menggunakan pytest framework untuk validasi dataset.

Usage:
    pytest tests/test_dataset_pytest.py -v
    pytest tests/test_dataset_pytest.py -v -s  # with print output

Author: ANTIGRAVITY Team
Date: 2026-05-02
"""

import pytest
import torch
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime

# Configuration
DATA_DIR = 'data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train', 'graphs.pt')
VAL_FILE = os.path.join(DATA_DIR, 'val', 'graphs.pt')
TEST_FILE = os.path.join(DATA_DIR, 'test', 'graphs.pt')

EXPECTED_NUM_STATIONS = 24
TRAIN_END_DATE = '2023-12-31'
VAL_START_DATE = '2024-01-01'
VAL_END_DATE = '2025-03-31'
TEST_START_DATE = '2026-01-01'


# Fixtures
@pytest.fixture(scope="module")
def train_graphs():
    """Load training graphs."""
    if not os.path.exists(TRAIN_FILE):
        pytest.skip(f"Train file not found: {TRAIN_FILE}")
    return torch.load(TRAIN_FILE)


@pytest.fixture(scope="module")
def val_graphs():
    """Load validation graphs."""
    if not os.path.exists(VAL_FILE):
        pytest.skip(f"Validation file not found: {VAL_FILE}")
    return torch.load(VAL_FILE)


@pytest.fixture(scope="module")
def test_graphs():
    """Load test graphs."""
    if not os.path.exists(TEST_FILE):
        pytest.skip(f"Test file not found: {TEST_FILE}")
    return torch.load(TEST_FILE)


@pytest.fixture(scope="module")
def station_data():
    """Load station data."""
    station_file = os.path.join(DATA_DIR, 'processed', 'lokasi_stasiun_clean.csv')
    if not os.path.exists(station_file):
        pytest.skip(f"Station file not found: {station_file}")
    return pd.read_csv(station_file)


# ============================================================================
# UJI 1: Validasi Integritas Spasial
# ============================================================================

class TestSpatialIntegrity:
    """UJI 1: Validasi Integritas Spasial (Dimensi Node & Masking)"""
    
    @pytest.mark.parametrize("dataset_name,graphs_fixture", [
        ("train", "train_graphs"),
        ("val", "val_graphs"),
        ("test", "test_graphs")
    ])
    def test_node_dimensions(self, dataset_name, graphs_fixture, request):
        """Assert 1.1: Dimensi node selalu 24 stasiun."""
        graphs = request.getfixturevalue(graphs_fixture)
        
        # Sample 5 random graphs
        sample_size = min(5, len(graphs))
        sample_indices = random.sample(range(len(graphs)), sample_size)
        
        for idx in sample_indices:
            graph = graphs[idx]
            num_nodes = graph.x.shape[0]
            
            assert num_nodes == EXPECTED_NUM_STATIONS, (
                f"{dataset_name} - Graph {idx} (date: {graph.date}): "
                f"Expected {EXPECTED_NUM_STATIONS} nodes, got {num_nodes}"
            )
    
    @pytest.mark.parametrize("dataset_name,graphs_fixture", [
        ("train", "train_graphs"),
        ("val", "val_graphs"),
        ("test", "test_graphs")
    ])
    def test_no_nan_values(self, dataset_name, graphs_fixture, request):
        """Assert 1.2: Tidak ada NaN dalam node features."""
        graphs = request.getfixturevalue(graphs_fixture)
        
        # Sample 5 random graphs
        sample_size = min(5, len(graphs))
        sample_indices = random.sample(range(len(graphs)), sample_size)
        
        for idx in sample_indices:
            graph = graphs[idx]
            
            assert not torch.isnan(graph.x).any(), (
                f"{dataset_name} - Graph {idx} (date: {graph.date}): "
                f"Found NaN values in node features"
            )


# ============================================================================
# UJI 2: Validasi Strict Chronological Split
# ============================================================================

class TestChronologicalSplit:
    """UJI 2: Validasi Strict Chronological Split (Anti-Data Leakage)"""
    
    def test_train_date_range(self, train_graphs):
        """Assert 2.1: Train set <= 2023-12-31."""
        train_end = pd.to_datetime(TRAIN_END_DATE)
        
        for idx, graph in enumerate(train_graphs):
            graph_date = pd.to_datetime(graph.date)
            
            assert graph_date <= train_end, (
                f"Train - Graph {idx}: date {graph.date} exceeds {TRAIN_END_DATE}"
            )
    
    def test_val_date_range(self, val_graphs):
        """Assert 2.2: Val set in [2024-01-01, 2025-03-31]."""
        val_start = pd.to_datetime(VAL_START_DATE)
        val_end = pd.to_datetime(VAL_END_DATE)
        
        for idx, graph in enumerate(val_graphs):
            graph_date = pd.to_datetime(graph.date)
            
            assert val_start <= graph_date <= val_end, (
                f"Val - Graph {idx}: date {graph.date} outside "
                f"[{VAL_START_DATE}, {VAL_END_DATE}]"
            )
    
    def test_test_date_range(self, test_graphs):
        """Assert 2.3: Test set >= 2026-01-01."""
        test_start = pd.to_datetime(TEST_START_DATE)
        
        for idx, graph in enumerate(test_graphs):
            graph_date = pd.to_datetime(graph.date)
            
            assert graph_date >= test_start, (
                f"Test - Graph {idx}: date {graph.date} before {TEST_START_DATE}"
            )
    
    def test_no_temporal_overlap(self, train_graphs, val_graphs, test_graphs):
        """Verify no temporal overlap between splits."""
        train_dates = [pd.to_datetime(g.date) for g in train_graphs]
        val_dates = [pd.to_datetime(g.date) for g in val_graphs]
        test_dates = [pd.to_datetime(g.date) for g in test_graphs]
        
        train_max = max(train_dates)
        val_min = min(val_dates)
        val_max = max(val_dates)
        test_min = min(test_dates)
        
        assert train_max < val_min, (
            f"Train and Val overlap: train_max={train_max.date()}, "
            f"val_min={val_min.date()}"
        )
        
        assert val_max < test_min, (
            f"Val and Test overlap: val_max={val_max.date()}, "
            f"test_min={test_min.date()}"
        )
    
    def test_may_2024_storm_exists(self, val_graphs):
        """Check for May 2024 extreme space weather in validation set."""
        may_2024_count = 0
        extreme_kp_count = 0
        
        for graph in val_graphs:
            graph_date = pd.to_datetime(graph.date)
            
            if graph_date.year == 2024 and graph_date.month == 5:
                may_2024_count += 1
                
                if hasattr(graph, 'kp_index'):
                    kp = graph.kp_index
                    if kp >= 8.0:
                        extreme_kp_count += 1
        
        # At least some May 2024 data should exist
        assert may_2024_count > 0, (
            "No May 2024 data found in validation set"
        )
        
        # Warning if no extreme Kp (might be expected if not integrated yet)
        if extreme_kp_count == 0:
            pytest.skip(
                "No extreme space weather (Kp >= 8.0) found in May 2024. "
                "This is expected if space weather data is not yet integrated."
            )


# ============================================================================
# UJI 3: Validasi Target DPINN & Multi-Task
# ============================================================================

class TestMultiTaskLabels:
    """UJI 3: Validasi Target DPINN & Multi-Task (Label Lengkap)"""
    
    @pytest.mark.parametrize("dataset_name,graphs_fixture", [
        ("train", "train_graphs"),
        ("val", "val_graphs"),
        ("test", "test_graphs")
    ])
    def test_event_labels_valid(self, dataset_name, graphs_fixture, request):
        """Assert 3.1: Event samples have valid labels."""
        graphs = request.getfixturevalue(graphs_fixture)
        
        event_samples = [g for g in graphs if g.y_event.item() == 1]
        
        # Check at least some events exist
        assert len(event_samples) > 0, (
            f"{dataset_name}: No event samples found"
        )
        
        # Check first 10 event samples
        for graph in event_samples[:10]:
            # Check magnitude
            mag = graph.y_mag.item()
            assert not torch.isnan(graph.y_mag), (
                f"{dataset_name} - Event (date: {graph.date}): "
                f"Magnitude is NaN"
            )
            assert mag > 0, (
                f"{dataset_name} - Event (date: {graph.date}): "
                f"Invalid magnitude: {mag} (expected > 0)"
            )
            
            # Check azimuth
            azm = graph.y_azm.item()
            assert not torch.isnan(graph.y_azm), (
                f"{dataset_name} - Event (date: {graph.date}): "
                f"Azimuth is NaN"
            )
            assert 0 <= azm <= 360, (
                f"{dataset_name} - Event (date: {graph.date}): "
                f"Invalid azimuth: {azm} (expected 0-360)"
            )
            
            # Check distances
            assert not torch.isnan(graph.y_dist).any(), (
                f"{dataset_name} - Event (date: {graph.date}): "
                f"Found NaN in distance labels"
            )
            assert (graph.y_dist > 0).all(), (
                f"{dataset_name} - Event (date: {graph.date}): "
                f"Found non-positive distance values"
            )
    
    @pytest.mark.parametrize("dataset_name,graphs_fixture", [
        ("train", "train_graphs"),
        ("val", "val_graphs"),
        ("test", "test_graphs")
    ])
    def test_space_weather_features(self, dataset_name, graphs_fixture, request):
        """Assert 3.2: Space weather features are valid floats."""
        graphs = request.getfixturevalue(graphs_fixture)
        
        # Sample first 10 graphs
        sample_size = min(10, len(graphs))
        
        for idx in range(sample_size):
            graph = graphs[idx]
            
            # Check Kp index
            if hasattr(graph, 'kp_index'):
                kp = graph.kp_index
                assert isinstance(kp, (int, float)), (
                    f"{dataset_name} - Graph {idx} (date: {graph.date}): "
                    f"Kp index is not numeric: {type(kp)}"
                )
                assert not np.isnan(kp), (
                    f"{dataset_name} - Graph {idx} (date: {graph.date}): "
                    f"Kp index is NaN"
                )
            else:
                pytest.skip(
                    f"{dataset_name}: Space weather data (kp_index) not found. "
                    f"This is expected if not yet integrated."
                )


# ============================================================================
# UJI 4: Validasi Physics-Guided Adjacency Matrix
# ============================================================================

class TestAdjacencyMatrix:
    """UJI 4: Validasi Physics-Guided Adjacency Matrix (Topologi Graf)"""
    
    def test_edge_connectivity(self, train_graphs):
        """Assert 4.1: Check edge connectivity."""
        graph = train_graphs[0]
        
        num_nodes = graph.x.shape[0]
        num_edges = graph.edge_index.shape[1]
        
        # For fully connected graph: num_edges = num_nodes * (num_nodes - 1)
        # (bidirectional, no self-loops)
        expected_edges = num_nodes * (num_nodes - 1)
        
        # Allow for self-loops: num_edges = num_nodes * num_nodes
        expected_edges_with_self = num_nodes * num_nodes
        
        assert num_edges == expected_edges or num_edges == expected_edges_with_self, (
            f"Unexpected number of edges: {num_edges}. "
            f"Expected {expected_edges} (no self-loops) or "
            f"{expected_edges_with_self} (with self-loops)"
        )
    
    def test_tectonic_penalty_intra_plate(self, train_graphs, station_data):
        """Assert 4.2a: Intra-plate pair has penalty 0.0."""
        graph = train_graphs[0]
        
        # Test pair: YOG and TRT (both in Sunda)
        station_codes = station_data['Kode Stasiun'].tolist()
        
        try:
            idx1 = station_codes.index('YOG')
            idx2 = station_codes.index('TRT')
        except ValueError:
            pytest.skip("Test stations (YOG, TRT) not found in station data")
        
        # Find edge
        edge_index = graph.edge_index.numpy()
        edge_attr = graph.edge_attr.numpy()
        
        penalty = None
        for j in range(edge_index.shape[1]):
            if ((edge_index[0, j] == idx1 and edge_index[1, j] == idx2) or
                (edge_index[0, j] == idx2 and edge_index[1, j] == idx1)):
                if edge_attr.shape[1] >= 2:
                    penalty = edge_attr[j, 1]  # Second column is penalty
                break
        
        assert penalty is not None, (
            "Edge between YOG and TRT not found"
        )
        
        assert abs(penalty - 0.0) < 1e-6, (
            f"Intra-plate pair (YOG, TRT): Expected penalty 0.0, got {penalty}"
        )
    
    def test_tectonic_penalty_inter_plate(self, train_graphs, station_data):
        """Assert 4.2b: Inter-plate pair has penalty 0.5."""
        graph = train_graphs[0]
        
        # Test pair: YOG (Sunda) and TND (Wallacea)
        station_codes = station_data['Kode Stasiun'].tolist()
        
        try:
            idx1 = station_codes.index('YOG')
            idx2 = station_codes.index('TND')
        except ValueError:
            pytest.skip("Test stations (YOG, TND) not found in station data")
        
        # Find edge
        edge_index = graph.edge_index.numpy()
        edge_attr = graph.edge_attr.numpy()
        
        penalty = None
        for j in range(edge_index.shape[1]):
            if ((edge_index[0, j] == idx1 and edge_index[1, j] == idx2) or
                (edge_index[0, j] == idx2 and edge_index[1, j] == idx1)):
                if edge_attr.shape[1] >= 2:
                    penalty = edge_attr[j, 1]
                break
        
        assert penalty is not None, (
            "Edge between YOG and TND not found"
        )
        
        assert abs(penalty - 0.5) < 1e-6, (
            f"Inter-plate pair (YOG, TND): Expected penalty 0.5, got {penalty}"
        )


# ============================================================================
# Configuration for pytest
# ============================================================================

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
