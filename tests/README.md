# ANTIGRAVITY Tests

Unit tests untuk validasi dataset dan model ANTIGRAVITY.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── test_dataset_pytest.py      # PyTest dataset validation
└── README.md                   # This file
```

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_dataset_pytest.py -v

# Run with output
pytest tests/ -v -s
```

### Test Categories

#### 1. Spatial Integrity Tests
```bash
pytest tests/test_dataset_pytest.py::TestSpatialIntegrity -v
```

Tests:
- Node dimensions (24 stations)
- No NaN values in features
- Proper masking for inactive stations

#### 2. Chronological Split Tests
```bash
pytest tests/test_dataset_pytest.py::TestChronologicalSplit -v
```

Tests:
- Train set date range (≤ 2023-12-31)
- Validation set date range (2024-01-01 to 2025-03-31)
- Test set date range (≥ 2026-01-01)
- No temporal overlap
- May 2024 storm presence

#### 3. Multi-Task Label Tests
```bash
pytest tests/test_dataset_pytest.py::TestMultiTaskLabels -v
```

Tests:
- Event labels validity (mag, azm, dist)
- Space weather features (Kp, Dst)
- No NaN in labels

#### 4. Adjacency Matrix Tests
```bash
pytest tests/test_dataset_pytest.py::TestAdjacencyMatrix -v
```

Tests:
- Edge connectivity
- Intra-plate tectonic penalty (0.0)
- Inter-plate tectonic penalty (0.5)

## Test Options

### Verbose Output
```bash
pytest tests/ -v
```

### Show Print Statements
```bash
pytest tests/ -v -s
```

### Run Specific Test
```bash
pytest tests/test_dataset_pytest.py::TestChronologicalSplit::test_train_date_range -v
```

### Stop on First Failure
```bash
pytest tests/ -x
```

### Run Last Failed Tests
```bash
pytest tests/ --lf
```

### Coverage Report
```bash
pytest tests/ --cov=scripts --cov-report=html
```

Then open `htmlcov/index.html` in browser.

### Parallel Execution
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run in parallel
pytest tests/ -n auto
```

## Expected Output

### All Tests Pass ✅
```
tests/test_dataset_pytest.py::TestSpatialIntegrity::test_node_dimensions[train] PASSED
tests/test_dataset_pytest.py::TestSpatialIntegrity::test_node_dimensions[val] PASSED
tests/test_dataset_pytest.py::TestSpatialIntegrity::test_node_dimensions[test] PASSED
tests/test_dataset_pytest.py::TestSpatialIntegrity::test_no_nan_values[train] PASSED
tests/test_dataset_pytest.py::TestSpatialIntegrity::test_no_nan_values[val] PASSED
tests/test_dataset_pytest.py::TestSpatialIntegrity::test_no_nan_values[test] PASSED
tests/test_dataset_pytest.py::TestChronologicalSplit::test_train_date_range PASSED
tests/test_dataset_pytest.py::TestChronologicalSplit::test_val_date_range PASSED
tests/test_dataset_pytest.py::TestChronologicalSplit::test_test_date_range PASSED
tests/test_dataset_pytest.py::TestChronologicalSplit::test_no_temporal_overlap PASSED
tests/test_dataset_pytest.py::TestMultiTaskLabels::test_event_labels_valid[train] PASSED
tests/test_dataset_pytest.py::TestMultiTaskLabels::test_event_labels_valid[val] PASSED
tests/test_dataset_pytest.py::TestMultiTaskLabels::test_event_labels_valid[test] PASSED
tests/test_dataset_pytest.py::TestMultiTaskLabels::test_space_weather_features[train] SKIPPED
tests/test_dataset_pytest.py::TestAdjacencyMatrix::test_edge_connectivity PASSED
tests/test_dataset_pytest.py::TestAdjacencyMatrix::test_tectonic_penalty_intra_plate PASSED
tests/test_dataset_pytest.py::TestAdjacencyMatrix::test_tectonic_penalty_inter_plate PASSED

========================= 16 passed, 1 skipped in 2.34s =========================
```

### Some Tests Fail ❌
```
tests/test_dataset_pytest.py::TestChronologicalSplit::test_train_date_range FAILED

================================= FAILURES =================================
______ TestChronologicalSplit.test_train_date_range ______

    def test_train_date_range(self, train_graphs):
        train_end = pd.to_datetime(TRAIN_END_DATE)
        
        for idx, graph in enumerate(train_graphs):
            graph_date = pd.to_datetime(graph.date)
            
>           assert graph_date <= train_end, (
                f"Train - Graph {idx}: date {graph.date} exceeds {TRAIN_END_DATE}"
            )
E           AssertionError: Train - Graph 1234: date 2024-01-15 exceeds 2023-12-31

tests/test_dataset_pytest.py:123: AssertionError
========================= 1 failed, 15 passed in 2.45s =========================
```

## Standalone Validation Script

For quick validation without pytest:

```bash
python scripts/test_dataset_validation.py
```

This script provides:
- More verbose output
- User-friendly summary
- Detailed error messages
- No pytest dependency required

## Fixtures

Tests use pytest fixtures for data loading:

- `train_graphs`: Training set graphs
- `val_graphs`: Validation set graphs
- `test_graphs`: Test set graphs
- `station_data`: Station metadata

Fixtures are loaded once per test session for efficiency.

## Markers

### Skip Tests
```python
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass
```

### Parametrize Tests
```python
@pytest.mark.parametrize("dataset_name,graphs_fixture", [
    ("train", "train_graphs"),
    ("val", "val_graphs"),
    ("test", "test_graphs")
])
def test_something(dataset_name, graphs_fixture, request):
    graphs = request.getfixturevalue(graphs_fixture)
    # Test logic
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run tests
  run: pytest tests/ -v --cov=scripts
```

### GitLab CI
```yaml
test:
  script:
    - pip install -r requirements.txt
    - pytest tests/ -v --cov=scripts
```

## Writing New Tests

### Test Template
```python
class TestNewFeature:
    """Test description."""
    
    def test_something(self, train_graphs):
        """Test specific behavior."""
        # Arrange
        graph = train_graphs[0]
        
        # Act
        result = some_function(graph)
        
        # Assert
        assert result == expected_value, "Error message"
```

### Best Practices
1. Use descriptive test names
2. One assertion per test (when possible)
3. Use fixtures for data loading
4. Add docstrings to tests
5. Use parametrize for similar tests
6. Keep tests independent

## Troubleshooting

### "No module named 'pytest'"
```bash
pip install pytest pytest-cov
```

### "File not found" errors
```bash
# Run from project root
cd /path/to/ANTIGRAVITY
pytest tests/ -v
```

### "Fixture not found"
```bash
# Make sure you're using correct fixture names
# Check tests/__init__.py exists
```

### Tests hang or timeout
```bash
# Add timeout
pytest tests/ -v --timeout=60
```

## Documentation

For detailed test documentation, see:
- `doc/06_dataset_validation_testing.md` - Complete testing guide
- `scripts/test_dataset_validation.py` - Standalone validation script

## Contact

For test-related issues:
1. Check test output for error messages
2. Review `doc/06_dataset_validation_testing.md`
3. Check logs in `data/processed/`
