# Dataset Validation & Unit Testing

**Tanggal**: 2 Mei 2026  
**Status**: Implementation Complete

## Tujuan

Melakukan pengujian otomatis (Automated Unit Testing) pada hasil pipeline dataset multi-stasiun untuk memvalidasi:
1. Integritas spasial (dimensi node & masking)
2. Strict chronological split (anti-data leakage)
3. Target DPINN & multi-task (label lengkap)
4. Physics-guided adjacency matrix (topologi graf)

## Testing Framework

Proyek ini menyediakan **2 cara** untuk menjalankan validasi dataset:

### 1. Standalone Script (Recommended untuk Quick Check)
```bash
python scripts/test_dataset_validation.py
```

**Keuntungan**:
- Tidak memerlukan pytest
- Output yang lebih verbose dan user-friendly
- Langsung menampilkan summary lengkap
- Cocok untuk debugging cepat

### 2. PyTest Framework (Recommended untuk CI/CD)
```bash
pytest tests/test_dataset_pytest.py -v
pytest tests/test_dataset_pytest.py -v -s  # with print output
```

**Keuntungan**:
- Integrasi dengan CI/CD pipeline
- Parallel test execution
- Coverage reporting
- Standard testing framework

## Kriteria Validasi

### UJI 1: Validasi Integritas Spasial

#### Assert 1.1: Dimensi Node
**Tujuan**: Memastikan setiap graph snapshot memiliki tepat 24 nodes (stasiun)

**Test**:
```python
for graph in sample_graphs:
    assert graph.x.shape[0] == 24
```

**Expected**: Semua graph memiliki 24 nodes

**Failure Indication**:
- Graph dengan jumlah node ≠ 24
- Menunjukkan error dalam graph construction

#### Assert 1.2: No NaN Values
**Tujuan**: Memastikan tidak ada NaN dalam node features

**Test**:
```python
for graph in sample_graphs:
    assert not torch.isnan(graph.x).any()
```

**Expected**: Tidak ada NaN values

**Failure Indication**:
- NaN dalam node features
- Stasiun offline tidak di-handle dengan benar
- Harus menggunakan zero-padding atau masking

### UJI 2: Validasi Strict Chronological Split

#### Assert 2.1: Train Date Range
**Tujuan**: Memastikan train set hanya berisi data ≤ 2023-12-31

**Test**:
```python
for graph in train_graphs:
    assert pd.to_datetime(graph.date) <= pd.to_datetime('2023-12-31')
```

**Expected**: Semua train graphs ≤ 2023-12-31

**Failure Indication**:
- **CRITICAL DATA LEAKAGE**: Data dari 2024+ masuk ke train set
- Model akan "melihat" masa depan
- Hasil evaluasi tidak valid

#### Assert 2.2: Validation Date Range
**Tujuan**: Memastikan val set dalam range [2024-01-01, 2025-03-31]

**Test**:
```python
for graph in val_graphs:
    date = pd.to_datetime(graph.date)
    assert pd.to_datetime('2024-01-01') <= date <= pd.to_datetime('2025-03-31')
```

**Expected**: Semua val graphs dalam range

**Failure Indication**:
- Data di luar range validation
- Potential data leakage

#### Assert 2.3: Test Date Range
**Tujuan**: Memastikan test set hanya berisi data ≥ 2026-01-01

**Test**:
```python
for graph in test_graphs:
    assert pd.to_datetime(graph.date) >= pd.to_datetime('2026-01-01')
```

**Expected**: Semua test graphs ≥ 2026-01-01

**Failure Indication**:
- **CRITICAL DATA LEAKAGE**: Data dari 2024-2025 masuk ke test set
- Blind test tidak valid

#### Assert 2.4: May 2024 Storm
**Tujuan**: Memastikan badai matahari ekstrem (Mei 2024) ada di validation set

**Test**:
```python
may_2024_storms = [
    g for g in val_graphs 
    if pd.to_datetime(g.date).year == 2024 
    and pd.to_datetime(g.date).month == 5
    and g.kp_index >= 8.0
]
assert len(may_2024_storms) > 0
```

**Expected**: Minimal 1 event dengan Kp ≥ 8.0 di Mei 2024

**Failure Indication**:
- Space weather data belum diintegrasikan (expected jika belum implementasi)
- Data Mei 2024 tidak ada di validation set

### UJI 3: Validasi Target DPINN & Multi-Task

#### Assert 3.1: Event Labels Valid
**Tujuan**: Memastikan event samples memiliki label yang valid

**Test**:
```python
for graph in event_samples:
    # Magnitude
    assert graph.y_mag.item() > 0
    assert not torch.isnan(graph.y_mag)
    
    # Azimuth
    azm = graph.y_azm.item()
    assert 0 <= azm <= 360
    assert not torch.isnan(graph.y_azm)
    
    # Distances
    assert (graph.y_dist > 0).all()
    assert not torch.isnan(graph.y_dist).any()
```

**Expected**: 
- Magnitude > 0
- Azimuth dalam range [0, 360]
- Distances > 0
- Tidak ada NaN

**Failure Indication**:
- Label extraction error
- Invalid earthquake data
- Missing distance calculation

#### Assert 3.2: Space Weather Features
**Tujuan**: Memastikan space weather features (Kp, Dst) valid

**Test**:
```python
for graph in sample_graphs:
    assert hasattr(graph, 'kp_index')
    assert isinstance(graph.kp_index, (int, float))
    assert not np.isnan(graph.kp_index)
```

**Expected**: Kp dan Dst adalah float yang valid

**Failure Indication**:
- Space weather data belum diintegrasikan (expected jika belum implementasi)
- Invalid space weather values

### UJI 4: Validasi Physics-Guided Adjacency Matrix

#### Assert 4.1: Edge Connectivity
**Tujuan**: Memastikan graph connectivity sesuai desain

**Test**:
```python
num_nodes = 24
num_edges = graph.edge_index.shape[1]

# Fully connected (no self-loops): 24 * 23 = 552
# Fully connected (with self-loops): 24 * 24 = 576
assert num_edges in [552, 576]
```

**Expected**: 552 edges (no self-loops) atau 576 edges (with self-loops)

**Failure Indication**:
- Graph tidak fully connected
- Missing edges
- Adjacency matrix construction error

#### Assert 4.2a: Intra-Plate Penalty
**Tujuan**: Memastikan stasiun dalam region yang sama memiliki penalty 0.0

**Test**:
```python
# YOG dan TRT (both in Sunda)
penalty = get_edge_penalty(graph, 'YOG', 'TRT')
assert abs(penalty - 0.0) < 1e-6
```

**Expected**: Penalty = 0.0

**Failure Indication**:
- Tectonic penalty tidak diterapkan dengan benar
- Region assignment error

#### Assert 4.2b: Inter-Plate Penalty
**Tujuan**: Memastikan stasiun di region berbeda memiliki penalty 0.5

**Test**:
```python
# YOG (Sunda) dan TND (Wallacea)
penalty = get_edge_penalty(graph, 'YOG', 'TND')
assert abs(penalty - 0.5) < 1e-6
```

**Expected**: Penalty = 0.5

**Failure Indication**:
- Tectonic penalty tidak diterapkan dengan benar
- Cross-tectonic edges tidak terdeteksi

## Running Tests

### Quick Validation (Standalone Script)

```bash
# Run all tests
python scripts/test_dataset_validation.py

# Expected output:
# [PASSED] Load Train Set
# [PASSED] Load Validation Set
# [PASSED] Load Test Set
# [PASSED] UJI 1.1 - Node Dimension
# [PASSED] UJI 1.2 - NaN Check
# ...
# ✅ ALL TESTS PASSED - Dataset is valid for training!
```

### Comprehensive Testing (PyTest)

```bash
# Run all tests with verbose output
pytest tests/test_dataset_pytest.py -v

# Run specific test class
pytest tests/test_dataset_pytest.py::TestSpatialIntegrity -v

# Run specific test
pytest tests/test_dataset_pytest.py::TestChronologicalSplit::test_train_date_range -v

# Run with coverage
pytest tests/test_dataset_pytest.py --cov=scripts --cov-report=html

# Run in parallel (requires pytest-xdist)
pytest tests/test_dataset_pytest.py -n auto
```

### Expected Output (PyTest)

```
tests/test_dataset_pytest.py::TestSpatialIntegrity::test_node_dimensions[train] PASSED
tests/test_dataset_pytest.py::TestSpatialIntegrity::test_node_dimensions[val] PASSED
tests/test_dataset_pytest.py::TestSpatialIntegrity::test_node_dimensions[test] PASSED
tests/test_dataset_pytest.py::TestSpatialIntegrity::test_no_nan_values[train] PASSED
tests/test_dataset_pytest.py::TestChronologicalSplit::test_train_date_range PASSED
tests/test_dataset_pytest.py::TestChronologicalSplit::test_val_date_range PASSED
tests/test_dataset_pytest.py::TestChronologicalSplit::test_test_date_range PASSED
tests/test_dataset_pytest.py::TestChronologicalSplit::test_no_temporal_overlap PASSED
tests/test_dataset_pytest.py::TestMultiTaskLabels::test_event_labels_valid[train] PASSED
tests/test_dataset_pytest.py::TestAdjacencyMatrix::test_tectonic_penalty_intra_plate PASSED
tests/test_dataset_pytest.py::TestAdjacencyMatrix::test_tectonic_penalty_inter_plate PASSED

========================= 15 passed in 2.34s =========================
```

## Interpreting Results

### All Tests Passed ✅
```
✅ ALL TESTS PASSED - Dataset is valid for training!
```

**Action**: Proceed to model training

### Some Tests Failed ❌
```
❌ SOME TESTS FAILED - Please fix issues before training!

FAILED TESTS:
UJI 2.1 - Train Date Range:
  Graph 1234: date 2024-01-15 exceeds 2023-12-31
```

**Action**: 
1. Review failed test details
2. Check data processing scripts
3. Re-run data pipeline if needed
4. Fix issues before training

### Warnings ⚠️
```
[WARNING] UJI 3.2 - Space Weather
          Space weather data (kp_index) not found.
          This is expected if not yet integrated.
```

**Action**: 
- Warnings are informational
- May indicate missing optional features
- Training can proceed if only warnings (no failures)

## Common Issues & Solutions

### Issue 1: "File not found"
**Symptom**: Cannot load train/val/test graphs

**Solution**:
```bash
# Run data processing pipeline
python scripts/01_data_cleaning.py
python scripts/02_graph_construction.py
python scripts/03_chronological_split.py
```

### Issue 2: "Data leakage detected"
**Symptom**: Train set contains dates > 2023-12-31

**Solution**:
- Check `scripts/03_chronological_split.py`
- Verify split dates in config
- Re-run chronological split

### Issue 3: "Invalid labels"
**Symptom**: Event samples have NaN or invalid values

**Solution**:
- Check `scripts/02_graph_construction.py`
- Verify earthquake catalog data
- Check label extraction logic

### Issue 4: "Tectonic penalty incorrect"
**Symptom**: Penalty values don't match expected (0.0 or 0.5)

**Solution**:
- Check station region assignment
- Verify adjacency matrix construction
- Check edge attribute calculation

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Dataset Validation

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run dataset validation
      run: |
        pytest tests/test_dataset_pytest.py -v --cov=scripts
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Best Practices

### Before Training
1. ✅ Run validation tests
2. ✅ Verify all tests pass
3. ✅ Review any warnings
4. ✅ Check dataset statistics

### During Development
1. ✅ Run tests after data pipeline changes
2. ✅ Add new tests for new features
3. ✅ Keep tests up-to-date with code

### Before Deployment
1. ✅ Run full test suite
2. ✅ Verify no data leakage
3. ✅ Check physics constraints
4. ✅ Validate on blind test set

## Test Coverage

Current test coverage:
- **Spatial Integrity**: 100%
- **Chronological Split**: 100%
- **Multi-Task Labels**: 100%
- **Adjacency Matrix**: 100%

**Overall**: 100% of critical validation criteria covered

## Next Steps

After all tests pass:
1. Review test summary
2. Check dataset statistics
3. Proceed to model training
4. Monitor training metrics

## Changelog

- **2026-05-02**: Initial implementation
  - Created standalone validation script
  - Created pytest test suite
  - Documented all test criteria
  - Added CI/CD integration guide

---

**Status**: ✅ Testing Framework Complete  
**Next Action**: Run tests after data processing  
**Documentation**: Complete
