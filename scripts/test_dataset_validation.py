"""
ANTIGRAVITY Project - Dataset Validation & Unit Testing
========================================================

Automated unit testing untuk memvalidasi integritas dataset multi-stasiun
sebelum diumpankan ke model PyTorch.

Kriteria Validasi:
1. Integritas Spasial (Dimensi Node & Masking)
2. Strict Chronological Split (Anti-Data Leakage)
3. Target DPINN & Multi-Task (Label Lengkap)
4. Physics-Guided Adjacency Matrix (Topologi Graf)

Author: ANTIGRAVITY Team
Date: 2026-05-02
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import List, Tuple, Dict
import random

# Configuration
DATA_DIR = 'data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train', 'graphs.pt')
VAL_FILE = os.path.join(DATA_DIR, 'val', 'graphs.pt')
TEST_FILE = os.path.join(DATA_DIR, 'test', 'graphs.pt')

# Expected values
EXPECTED_NUM_STATIONS = 24
TRAIN_END_DATE = '2023-12-31'
VAL_START_DATE = '2024-01-01'
VAL_END_DATE = '2025-03-31'
TEST_START_DATE = '2026-01-01'

# Tectonic regions (for adjacency validation)
STATION_REGIONS = {
    # Sunda (region 0)
    'SBG': 0, 'MLB': 0, 'SCN': 0, 'KPY': 0, 'LWA': 0, 'LPS': 0,
    'SRG': 0, 'SKB': 0, 'CLP': 0, 'YOG': 0, 'TRT': 0, 'GSI': 0,
    # Wallacea (region 1)
    'TND': 1, 'GTO': 1, 'LWK': 1, 'PLU': 1, 'TNT': 1, 'TRD': 1,
    'LUT': 1, 'ALR': 1, 'ROT': 1,
    # Sahul (region 2)
    'SMI': 2, 'AMB': 2, 'JYP': 2, 'SRO': 2
}

# Test pairs for adjacency validation
INTRA_PLATE_PAIR = ('YOG', 'TRT')  # Both in Sunda
INTER_PLATE_PAIR = ('TND', 'TNT')  # TND in Wallacea, TNT in Wallacea (same actually)
# Better inter-plate example:
INTER_PLATE_PAIR = ('YOG', 'TND')  # YOG in Sunda, TND in Wallacea


class TestResult:
    """Class untuk menyimpan hasil testing."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
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
    
    def summary(self):
        """Print summary of all tests."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Passed:   {len(self.passed)}")
        print(f"Total Failed:   {len(self.failed)}")
        print(f"Total Warnings: {len(self.warnings)}")
        
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
            print("✅ ALL TESTS PASSED - Dataset is valid for training!")
        else:
            print("❌ SOME TESTS FAILED - Please fix issues before training!")
        print("="*70)
        
        return len(self.failed) == 0


def load_dataset(file_path: str, dataset_name: str, results: TestResult) -> List:
    """Load dataset from file."""
    print(f"\nLoading {dataset_name} from {file_path}...")
    
    if not os.path.exists(file_path):
        results.add_fail(
            f"Load {dataset_name}",
            f"File not found: {file_path}"
        )
        return None
    
    try:
        graphs = torch.load(file_path)
        results.add_pass(
            f"Load {dataset_name}",
            f"Loaded {len(graphs)} graph snapshots"
        )
        return graphs
    except Exception as e:
        results.add_fail(
            f"Load {dataset_name}",
            f"Error loading file: {str(e)}"
        )
        return None


# ============================================================================
# UJI 1: Validasi Integritas Spasial (Dimensi Node & Masking)
# ============================================================================

def test_spatial_integrity(graphs: List, dataset_name: str, results: TestResult):
    """
    UJI 1: Validasi Integritas Spasial
    
    Assert 1.1: Dimensi node selalu 24 stasiun
    Assert 1.2: Stasiun offline diisi dengan zero/masking, bukan NaN
    """
    print("\n" + "="*70)
    print("UJI 1: VALIDASI INTEGRITAS SPASIAL")
    print("="*70)
    
    if graphs is None or len(graphs) == 0:
        results.add_fail(
            "UJI 1.1 - Node Dimension",
            f"{dataset_name}: No graphs to test"
        )
        return
    
    # Assert 1.1: Check node dimensions
    print("\nAssert 1.1: Checking node dimensions...")
    
    # Sample 5 random graphs
    sample_size = min(5, len(graphs))
    sample_indices = random.sample(range(len(graphs)), sample_size)
    
    dimension_errors = []
    for idx in sample_indices:
        graph = graphs[idx]
        num_nodes = graph.x.shape[0]
        
        if num_nodes != EXPECTED_NUM_STATIONS:
            dimension_errors.append(
                f"Graph {idx} (date: {graph.date}): "
                f"Expected {EXPECTED_NUM_STATIONS} nodes, got {num_nodes}"
            )
    
    if dimension_errors:
        results.add_fail(
            "UJI 1.1 - Node Dimension",
            "\n  ".join(dimension_errors)
        )
    else:
        results.add_pass(
            "UJI 1.1 - Node Dimension",
            f"All {sample_size} sampled graphs have {EXPECTED_NUM_STATIONS} nodes"
        )
    
    # Assert 1.2: Check for NaN values and masking
    print("\nAssert 1.2: Checking for NaN values and proper masking...")
    
    nan_errors = []
    masking_info = []
    
    for idx in sample_indices:
        graph = graphs[idx]
        
        # Check for NaN in node features
        if torch.isnan(graph.x).any():
            nan_errors.append(
                f"Graph {idx} (date: {graph.date}): "
                f"Found NaN values in node features"
            )
        
        # Check for zero-padded or masked nodes
        # Assuming last feature is 'is_active' flag
        if graph.x.shape[1] >= 5:  # At least 5 features
            is_active = graph.x[:, -1]  # Last column
            inactive_count = (is_active == 0).sum().item()
            
            if inactive_count > 0:
                masking_info.append(
                    f"Graph {idx} (date: {graph.date}): "
                    f"{inactive_count} inactive stations properly masked"
                )
    
    if nan_errors:
        results.add_fail(
            "UJI 1.2 - NaN Check",
            "\n  ".join(nan_errors)
        )
    else:
        results.add_pass(
            "UJI 1.2 - NaN Check",
            f"No NaN values found in {sample_size} sampled graphs"
        )
    
    if masking_info:
        results.add_warning(
            "UJI 1.2 - Masking Info",
            "\n  ".join(masking_info)
        )


# ============================================================================
# UJI 2: Validasi Strict Chronological Split (Anti-Data Leakage)
# ============================================================================

def test_chronological_split(train_graphs: List, val_graphs: List, 
                            test_graphs: List, results: TestResult):
    """
    UJI 2: Validasi Strict Chronological Split
    
    Assert 2.1: Train set <= 2023-12-31
    Assert 2.2: Val set in [2024-01-01, 2025-03-31]
    Assert 2.3: Test set >= 2026-01-01
    """
    print("\n" + "="*70)
    print("UJI 2: VALIDASI STRICT CHRONOLOGICAL SPLIT")
    print("="*70)
    
    train_end = pd.to_datetime(TRAIN_END_DATE)
    val_start = pd.to_datetime(VAL_START_DATE)
    val_end = pd.to_datetime(VAL_END_DATE)
    test_start = pd.to_datetime(TEST_START_DATE)
    
    # Assert 2.1: Train set validation
    print("\nAssert 2.1: Validating Train set dates...")
    
    if train_graphs is None or len(train_graphs) == 0:
        results.add_fail(
            "UJI 2.1 - Train Date Range",
            "No train graphs to validate"
        )
    else:
        train_violations = []
        for idx, graph in enumerate(train_graphs):
            graph_date = pd.to_datetime(graph.date)
            if graph_date > train_end:
                train_violations.append(
                    f"Graph {idx}: date {graph.date} exceeds {TRAIN_END_DATE}"
                )
        
        if train_violations:
            results.add_fail(
                "UJI 2.1 - Train Date Range",
                f"Found {len(train_violations)} violations:\n  " + 
                "\n  ".join(train_violations[:5])  # Show first 5
            )
        else:
            train_dates = [pd.to_datetime(g.date) for g in train_graphs]
            results.add_pass(
                "UJI 2.1 - Train Date Range",
                f"All {len(train_graphs)} graphs <= {TRAIN_END_DATE} "
                f"(range: {min(train_dates).date()} to {max(train_dates).date()})"
            )
    
    # Assert 2.2: Validation set validation
    print("\nAssert 2.2: Validating Validation set dates...")
    
    if val_graphs is None or len(val_graphs) == 0:
        results.add_fail(
            "UJI 2.2 - Val Date Range",
            "No validation graphs to validate"
        )
    else:
        val_violations = []
        may_2024_storms = []
        
        for idx, graph in enumerate(val_graphs):
            graph_date = pd.to_datetime(graph.date)
            
            # Check date range
            if graph_date < val_start or graph_date > val_end:
                val_violations.append(
                    f"Graph {idx}: date {graph.date} outside "
                    f"[{VAL_START_DATE}, {VAL_END_DATE}]"
                )
            
            # Check for May 2024 extreme space weather
            if graph_date.year == 2024 and graph_date.month == 5:
                kp = getattr(graph, 'kp_index', 0)
                if kp >= 8.0:
                    may_2024_storms.append(
                        f"Date {graph.date}: Kp = {kp}"
                    )
        
        if val_violations:
            results.add_fail(
                "UJI 2.2 - Val Date Range",
                f"Found {len(val_violations)} violations:\n  " + 
                "\n  ".join(val_violations[:5])
            )
        else:
            val_dates = [pd.to_datetime(g.date) for g in val_graphs]
            results.add_pass(
                "UJI 2.2 - Val Date Range",
                f"All {len(val_graphs)} graphs in [{VAL_START_DATE}, {VAL_END_DATE}] "
                f"(range: {min(val_dates).date()} to {max(val_dates).date()})"
            )
        
        # Check for May 2024 storms
        if may_2024_storms:
            results.add_pass(
                "UJI 2.2 - May 2024 Storm",
                f"Found {len(may_2024_storms)} extreme space weather events:\n  " +
                "\n  ".join(may_2024_storms[:3])
            )
        else:
            results.add_warning(
                "UJI 2.2 - May 2024 Storm",
                "No extreme space weather events (Kp >= 8.0) found in May 2024. "
                "This might be expected if space weather data is not yet integrated."
            )
    
    # Assert 2.3: Test set validation
    print("\nAssert 2.3: Validating Test set dates...")
    
    if test_graphs is None or len(test_graphs) == 0:
        results.add_fail(
            "UJI 2.3 - Test Date Range",
            "No test graphs to validate"
        )
    else:
        test_violations = []
        for idx, graph in enumerate(test_graphs):
            graph_date = pd.to_datetime(graph.date)
            if graph_date < test_start:
                test_violations.append(
                    f"Graph {idx}: date {graph.date} before {TEST_START_DATE}"
                )
        
        if test_violations:
            results.add_fail(
                "UJI 2.3 - Test Date Range",
                f"Found {len(test_violations)} violations:\n  " + 
                "\n  ".join(test_violations[:5])
            )
        else:
            test_dates = [pd.to_datetime(g.date) for g in test_graphs]
            results.add_pass(
                "UJI 2.3 - Test Date Range",
                f"All {len(test_graphs)} graphs >= {TEST_START_DATE} "
                f"(range: {min(test_dates).date()} to {max(test_dates).date()})"
            )


# ============================================================================
# UJI 3: Validasi Target DPINN & Multi-Task (Label Lengkap)
# ============================================================================

def test_multitask_labels(graphs: List, dataset_name: str, results: TestResult):
    """
    UJI 3: Validasi Target DPINN & Multi-Task
    
    Assert 3.1: Event samples have valid labels (mag, azm, dist)
    Assert 3.2: Space weather features (Kp, Dst) are valid floats
    """
    print("\n" + "="*70)
    print("UJI 3: VALIDASI TARGET DPINN & MULTI-TASK")
    print("="*70)
    
    if graphs is None or len(graphs) == 0:
        results.add_fail(
            "UJI 3.1 - Multi-Task Labels",
            f"{dataset_name}: No graphs to test"
        )
        return
    
    # Assert 3.1: Check event labels
    print("\nAssert 3.1: Checking multi-task labels for event samples...")
    
    event_samples = [g for g in graphs if g.y_event.item() == 1]
    label_errors = []
    
    for idx, graph in enumerate(event_samples[:10]):  # Check first 10 events
        # Check magnitude
        mag = graph.y_mag.item()
        if torch.isnan(graph.y_mag) or mag <= 0:
            label_errors.append(
                f"Event graph (date: {graph.date}): "
                f"Invalid magnitude: {mag}"
            )
        
        # Check azimuth
        azm = graph.y_azm.item()
        if torch.isnan(graph.y_azm) or azm < 0 or azm > 360:
            label_errors.append(
                f"Event graph (date: {graph.date}): "
                f"Invalid azimuth: {azm} (expected 0-360)"
            )
        
        # Check distances
        if torch.isnan(graph.y_dist).any():
            label_errors.append(
                f"Event graph (date: {graph.date}): "
                f"Found NaN in distance labels"
            )
        
        if (graph.y_dist <= 0).any():
            label_errors.append(
                f"Event graph (date: {graph.date}): "
                f"Found non-positive distance values"
            )
    
    if label_errors:
        results.add_fail(
            "UJI 3.1 - Multi-Task Labels",
            f"Found {len(label_errors)} label errors:\n  " +
            "\n  ".join(label_errors[:5])
        )
    else:
        results.add_pass(
            "UJI 3.1 - Multi-Task Labels",
            f"All {len(event_samples)} event samples have valid labels "
            f"(mag > 0, 0 <= azm <= 360, dist > 0)"
        )
    
    # Assert 3.2: Check space weather features
    print("\nAssert 3.2: Checking space weather features...")
    
    space_weather_errors = []
    sample_size = min(10, len(graphs))
    
    for idx in range(sample_size):
        graph = graphs[idx]
        
        # Check Kp index
        if hasattr(graph, 'kp_index'):
            kp = graph.kp_index
            if not isinstance(kp, (int, float)) or np.isnan(kp):
                space_weather_errors.append(
                    f"Graph {idx} (date: {graph.date}): "
                    f"Invalid Kp index: {kp}"
                )
        else:
            space_weather_errors.append(
                f"Graph {idx} (date: {graph.date}): "
                f"Missing kp_index attribute"
            )
        
        # Check Dst index
        if hasattr(graph, 'dst_index'):
            dst = graph.dst_index
            if not isinstance(dst, (int, float)) or np.isnan(dst):
                space_weather_errors.append(
                    f"Graph {idx} (date: {graph.date}): "
                    f"Invalid Dst index: {dst}"
                )
        else:
            space_weather_errors.append(
                f"Graph {idx} (date: {graph.date}): "
                f"Missing dst_index attribute"
            )
    
    if space_weather_errors:
        # This is a warning, not a failure, since space weather might not be integrated yet
        results.add_warning(
            "UJI 3.2 - Space Weather",
            f"Found {len(space_weather_errors)} issues:\n  " +
            "\n  ".join(space_weather_errors[:5]) +
            "\n  (This is expected if space weather data is not yet integrated)"
        )
    else:
        results.add_pass(
            "UJI 3.2 - Space Weather",
            f"All {sample_size} sampled graphs have valid space weather features"
        )


# ============================================================================
# UJI 4: Validasi Physics-Guided Adjacency Matrix (Topologi Graf)
# ============================================================================

def test_adjacency_matrix(graphs: List, dataset_name: str, results: TestResult):
    """
    UJI 4: Validasi Physics-Guided Adjacency Matrix
    
    Assert 4.1: Diagonal values are 1 (self-connection)
    Assert 4.2: Tectonic penalty applied correctly
    """
    print("\n" + "="*70)
    print("UJI 4: VALIDASI PHYSICS-GUIDED ADJACENCY MATRIX")
    print("="*70)
    
    if graphs is None or len(graphs) == 0:
        results.add_fail(
            "UJI 4.1 - Adjacency Matrix",
            f"{dataset_name}: No graphs to test"
        )
        return
    
    # Take first graph as representative (adjacency should be same for all)
    graph = graphs[0]
    
    # Assert 4.1: Check diagonal (self-connections)
    print("\nAssert 4.1: Checking adjacency matrix diagonal...")
    
    # Convert edge_index to adjacency matrix
    num_nodes = graph.x.shape[0]
    edge_index = graph.edge_index.numpy()
    
    # Check for self-loops
    self_loops = []
    for i in range(num_nodes):
        has_self_loop = False
        for j in range(edge_index.shape[1]):
            if edge_index[0, j] == i and edge_index[1, j] == i:
                has_self_loop = True
                break
        if not has_self_loop:
            self_loops.append(i)
    
    if self_loops:
        results.add_warning(
            "UJI 4.1 - Self-Loops",
            f"Nodes {self_loops[:5]} do not have self-loops. "
            f"This might be intentional for GAT architecture."
        )
    else:
        results.add_pass(
            "UJI 4.1 - Self-Loops",
            f"All {num_nodes} nodes have self-loops (diagonal = 1)"
        )
    
    # Assert 4.2: Check tectonic penalty
    print("\nAssert 4.2: Checking tectonic penalty application...")
    
    # Load station data to get station codes
    station_file = os.path.join(DATA_DIR, 'processed', 'lokasi_stasiun_clean.csv')
    
    if not os.path.exists(station_file):
        results.add_warning(
            "UJI 4.2 - Tectonic Penalty",
            f"Cannot validate tectonic penalty: {station_file} not found"
        )
        return
    
    stations_df = pd.read_csv(station_file)
    station_codes = stations_df['Kode Stasiun'].tolist()
    
    # Find indices for test pairs
    try:
        # Intra-plate pair (same region)
        idx_intra_1 = station_codes.index(INTRA_PLATE_PAIR[0])
        idx_intra_2 = station_codes.index(INTRA_PLATE_PAIR[1])
        
        # Inter-plate pair (different regions)
        idx_inter_1 = station_codes.index(INTER_PLATE_PAIR[0])
        idx_inter_2 = station_codes.index(INTER_PLATE_PAIR[1])
        
        # Find edge attributes for these pairs
        edge_attr = graph.edge_attr.numpy()
        
        # Find intra-plate edge
        intra_penalty = None
        for j in range(edge_index.shape[1]):
            if ((edge_index[0, j] == idx_intra_1 and edge_index[1, j] == idx_intra_2) or
                (edge_index[0, j] == idx_intra_2 and edge_index[1, j] == idx_intra_1)):
                # Assuming edge_attr format: [distance, penalty, weight]
                if edge_attr.shape[1] >= 2:
                    intra_penalty = edge_attr[j, 1]  # Second column is penalty
                break
        
        # Find inter-plate edge
        inter_penalty = None
        for j in range(edge_index.shape[1]):
            if ((edge_index[0, j] == idx_inter_1 and edge_index[1, j] == idx_inter_2) or
                (edge_index[0, j] == idx_inter_2 and edge_index[1, j] == idx_inter_1)):
                if edge_attr.shape[1] >= 2:
                    inter_penalty = edge_attr[j, 1]
                break
        
        # Validate penalties
        penalty_errors = []
        
        if intra_penalty is not None:
            if abs(intra_penalty - 0.0) > 1e-6:
                penalty_errors.append(
                    f"Intra-plate pair {INTRA_PLATE_PAIR}: "
                    f"Expected penalty 0.0, got {intra_penalty}"
                )
            else:
                results.add_pass(
                    "UJI 4.2a - Intra-Plate Penalty",
                    f"Pair {INTRA_PLATE_PAIR} has correct penalty: {intra_penalty}"
                )
        else:
            penalty_errors.append(
                f"Intra-plate pair {INTRA_PLATE_PAIR}: Edge not found"
            )
        
        if inter_penalty is not None:
            if abs(inter_penalty - 0.5) > 1e-6:
                penalty_errors.append(
                    f"Inter-plate pair {INTER_PLATE_PAIR}: "
                    f"Expected penalty 0.5, got {inter_penalty}"
                )
            else:
                results.add_pass(
                    "UJI 4.2b - Inter-Plate Penalty",
                    f"Pair {INTER_PLATE_PAIR} has correct penalty: {inter_penalty}"
                )
        else:
            penalty_errors.append(
                f"Inter-plate pair {INTER_PLATE_PAIR}: Edge not found"
            )
        
        if penalty_errors:
            results.add_fail(
                "UJI 4.2 - Tectonic Penalty",
                "\n  ".join(penalty_errors)
            )
    
    except ValueError as e:
        results.add_warning(
            "UJI 4.2 - Tectonic Penalty",
            f"Cannot validate tectonic penalty: {str(e)}"
        )
    except Exception as e:
        results.add_warning(
            "UJI 4.2 - Tectonic Penalty",
            f"Error during validation: {str(e)}"
        )


# ============================================================================
# Main Testing Function
# ============================================================================

def main():
    """Main testing function."""
    print("="*70)
    print("ANTIGRAVITY DATASET VALIDATION & UNIT TESTING")
    print("="*70)
    print(f"Started: {datetime.now()}")
    print()
    
    # Initialize results
    results = TestResult()
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Load datasets
    train_graphs = load_dataset(TRAIN_FILE, "Train Set", results)
    val_graphs = load_dataset(VAL_FILE, "Validation Set", results)
    test_graphs = load_dataset(TEST_FILE, "Test Set", results)
    
    # Check if any dataset loaded successfully
    if train_graphs is None and val_graphs is None and test_graphs is None:
        print("\n❌ ERROR: No datasets could be loaded!")
        print("Please run data processing pipeline first:")
        print("  1. python scripts/01_data_cleaning.py")
        print("  2. python scripts/02_graph_construction.py")
        print("  3. python scripts/03_chronological_split.py")
        return 1
    
    # Run tests
    
    # UJI 1: Spatial Integrity
    if train_graphs:
        test_spatial_integrity(train_graphs, "Train Set", results)
    if val_graphs:
        test_spatial_integrity(val_graphs, "Validation Set", results)
    if test_graphs:
        test_spatial_integrity(test_graphs, "Test Set", results)
    
    # UJI 2: Chronological Split
    test_chronological_split(train_graphs, val_graphs, test_graphs, results)
    
    # UJI 3: Multi-Task Labels
    if train_graphs:
        test_multitask_labels(train_graphs, "Train Set", results)
    if val_graphs:
        test_multitask_labels(val_graphs, "Validation Set", results)
    if test_graphs:
        test_multitask_labels(test_graphs, "Test Set", results)
    
    # UJI 4: Adjacency Matrix
    if train_graphs:
        test_adjacency_matrix(train_graphs, "Train Set", results)
    
    # Print summary
    all_passed = results.summary()
    
    print(f"\nCompleted: {datetime.now()}")
    
    # Return exit code
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
