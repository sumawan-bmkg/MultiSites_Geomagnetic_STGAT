# Changelog

All notable changes to the ANTIGRAVITY project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Model implementation (GAT + DPINN)
- Training script
- Evaluation script
- Visualization tools
- Space weather data integration
- Cosmic Gating module

## [0.1.1] - 2026-05-02

### Added
- **Dataset Validation & Unit Testing**:
  - `scripts/test_dataset_validation.py` - Standalone validation script
  - `tests/test_dataset_pytest.py` - PyTest test suite
  - `tests/__init__.py` - Test package initialization
  - `tests/README.md` - Test documentation
  - `doc/06_dataset_validation_testing.md` - Complete testing guide
- **Test Coverage**:
  - UJI 1: Validasi Integritas Spasial (Node dimensions, NaN check)
  - UJI 2: Validasi Strict Chronological Split (Anti-data leakage)
  - UJI 3: Validasi Target DPINN & Multi-Task (Label validation)
  - UJI 4: Validasi Physics-Guided Adjacency Matrix (Tectonic penalty)
- **Testing Dependencies**:
  - pytest>=7.3.0
  - pytest-cov>=4.1.0

### Features
- Automated validation of 4 critical dataset criteria
- Both standalone and pytest-based testing options
- Comprehensive error reporting with specific failure details
- CI/CD integration ready
- 100% coverage of critical validation criteria

## [0.1.0] - 2026-05-02

### Added
- Initial project structure
- Complete documentation suite:
  - `doc/00_project_overview.md` - Project overview and goals
  - `doc/01_data_cleaning.md` - Data cleaning procedures
  - `doc/02_graph_construction.md` - Graph construction methodology
  - `doc/03_physics_informed_loss.md` - Physics-informed loss design
  - `doc/04_chronological_split.md` - Chronological split strategy
  - `doc/05_model_architecture.md` - Model architecture design
- Data processing scripts:
  - `scripts/01_data_cleaning.py` - Clean raw data
  - `scripts/02_graph_construction.py` - Build graph snapshots
  - `scripts/03_chronological_split.py` - Split dataset chronologically
- Configuration files:
  - `config/model_config.yaml` - Model and training configuration
- Project files:
  - `README.md` - Project overview and quick start
  - `requirements.txt` - Python dependencies
  - `data/README.md` - Data directory documentation
  - `.gitignore` - Git ignore rules
- Directory structure:
  - `data/` - Data storage (raw, processed, train, val, test)
  - `models/` - Model storage
  - `scripts/` - Processing scripts
  - `config/` - Configuration files
  - `plots/` - Visualization outputs
  - `references/` - Research papers
  - `doc/` - Documentation
  - `notebooks/` - Jupyter notebooks

### Design Decisions
- **Multi-Station Simultaneous Processing**: Each graph snapshot represents all 24 stations on a single day
- **Strict Chronological Split**: Train (≤2023), Val (2024-2025), Test (≥2026) to prevent data leakage
- **Tectonic-Aware Adjacency**: Edge penalties for cross-tectonic connections
- **Physics-Informed Loss**: Attenuation law A = A₀ × e^(-α×d) integrated into training
- **Multi-Task Learning**: Simultaneous prediction of event, magnitude, azimuth, and distance

### Technical Specifications
- **Stations**: 24 seismik stations across Indonesia
- **Tectonic Regions**: Sunda (12), Wallacea (9), Sahul (3)
- **Prekursor Window**: 14 days before significant earthquakes
- **Significant Magnitude**: Mw ≥ 5.0
- **Graph Type**: Fully connected (276 edges per snapshot)
- **Node Features**: 5 basic features (lat, lon, elev, region, active)
- **Edge Features**: 3 features (distance, penalty, weight)

### Dependencies
- PyTorch ≥ 2.0.0
- PyTorch Geometric ≥ 2.3.0
- pandas ≥ 2.0.0
- numpy ≥ 1.24.0
- scipy ≥ 1.10.0
- h5py ≥ 3.8.0
- scikit-learn ≥ 1.3.0
- matplotlib ≥ 3.7.0
- seaborn ≥ 0.12.0

### Known Issues
- Raw data files not yet available in workspace
- Space weather data integration pending
- Model implementation pending
- Training pipeline pending

### Notes
- Project initialized on 2026-05-02
- Awaiting raw data files:
  - `data/raw/lokasi_stasiun.csv`
  - `data/raw/earthquake_catalog_2018_2025_merged_robust.csv`

## Version History

### Version Numbering
- **Major version** (X.0.0): Significant architectural changes
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, documentation updates

### Milestones
- **v0.1.0**: Project initialization and documentation
- **v0.2.0** (Planned): Data processing pipeline complete
- **v0.3.0** (Planned): Model implementation complete
- **v0.4.0** (Planned): Training pipeline complete
- **v0.5.0** (Planned): Evaluation and visualization tools
- **v1.0.0** (Planned): First production-ready release

## Contributing

When contributing to this project:
1. Update this CHANGELOG.md with your changes
2. Follow the format: Added, Changed, Deprecated, Removed, Fixed, Security
3. Include date and version number
4. Reference issue numbers when applicable

## References

### Key Papers
1. Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
2. Veličković et al. (2018) - Graph Attention Networks
3. Raissi et al. (2019) - Physics-Informed Neural Networks
4. Wu et al. (2020) - A Comprehensive Survey on Graph Neural Networks
5. Battaglia et al. (2018) - Relational inductive biases, deep learning, and graph networks

### Related Work
- Earthquake prediction using machine learning
- Seismic precursor detection
- Spatio-temporal graph neural networks
- Physics-informed deep learning
- Space weather effects on seismic signals

---

**Last Updated**: 2026-05-02  
**Project Status**: Initialization Phase  
**Next Milestone**: v0.2.0 - Data Processing Complete
