# Notebooks Directory

This directory contains Jupyter notebooks for data exploration, analysis, and experimentation.

## Notebook Categories

### 1. Data Exploration
- `01_explore_stations.ipynb` - Visualize station locations and metadata
- `02_explore_earthquakes.ipynb` - Analyze earthquake catalog
- `03_spatial_distribution.ipynb` - Spatial patterns and clustering
- `04_temporal_patterns.ipynb` - Temporal trends and seasonality

### 2. Graph Analysis
- `05_graph_structure.ipynb` - Analyze graph topology
- `06_adjacency_analysis.ipynb` - Edge weights and connectivity
- `07_node_features.ipynb` - Node feature distributions
- `08_graph_statistics.ipynb` - Graph-level statistics

### 3. Model Development
- `09_model_prototype.ipynb` - Initial model prototyping
- `10_loss_function_test.ipynb` - Test loss functions
- `11_physics_validation.ipynb` - Validate physics constraints
- `12_hyperparameter_search.ipynb` - Hyperparameter tuning

### 4. Results Analysis
- `13_training_analysis.ipynb` - Analyze training results
- `14_error_analysis.ipynb` - Deep dive into errors
- `15_physics_compliance.ipynb` - Check physics compliance
- `16_case_studies.ipynb` - Specific earthquake case studies

### 5. Visualization
- `17_interactive_maps.ipynb` - Interactive visualizations
- `18_animation.ipynb` - Temporal animations
- `19_publication_figures.ipynb` - High-quality figures for papers

## Setup

### Install Jupyter
```bash
pip install jupyter ipython ipykernel
```

### Register Kernel
```bash
python -m ipykernel install --user --name antigravity --display-name "ANTIGRAVITY"
```

### Launch Jupyter
```bash
jupyter notebook
# or
jupyter lab
```

## Notebook Template

Create new notebooks with this structure:

```python
# Notebook Title
# Description: Brief description of notebook purpose
# Author: Your name
# Date: YYYY-MM-DD

# ============================================================================
# Setup
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.data import Data

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# Load Data
# ============================================================================

# Load your data here

# ============================================================================
# Analysis
# ============================================================================

# Your analysis code here

# ============================================================================
# Visualization
# ============================================================================

# Your visualization code here

# ============================================================================
# Conclusions
# ============================================================================

# Summary of findings
```

## Best Practices

### 1. Organization
- One notebook per analysis topic
- Clear section headers with markdown
- Numbered cells for easy reference

### 2. Documentation
- Add markdown cells explaining each step
- Comment complex code
- Include references to papers/methods

### 3. Reproducibility
- Set random seeds
- Document package versions
- Save intermediate results

### 4. Visualization
- Use consistent color schemes
- Add titles, labels, and legends
- Save important figures to `plots/`

### 5. Performance
- Clear output before committing
- Don't commit large data files
- Use `%%time` to profile cells

## Example: Data Exploration

```python
# 01_explore_stations.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Load station data
stations = pd.read_csv('../data/processed/lokasi_stasiun_clean.csv')

# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(stations['Longitude'], stations['Latitude'])]
gdf = gpd.GeoDataFrame(stations, geometry=geometry, crs='EPSG:4326')

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(ax=ax, column='region_idx', cmap='viridis', 
         markersize=100, legend=True, alpha=0.7)

# Add labels
for idx, row in gdf.iterrows():
    ax.annotate(row['Kode Stasiun'], 
                xy=(row['Longitude'], row['Latitude']),
                xytext=(3, 3), textcoords='offset points',
                fontsize=8)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('24 Seismic Stations in Indonesia')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/notebook_station_map.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistics
print(f"Total stations: {len(stations)}")
print(f"\nStations by region:")
print(stations.groupby('region_idx')['Kode Stasiun'].count())
```

## Example: Graph Analysis

```python
# 05_graph_structure.ipynb

import torch
import networkx as nx
import matplotlib.pyplot as plt

# Load graphs
graphs = torch.load('../data/processed/dataset_graphs.pt')

# Take first graph as example
graph = graphs[0]

print(f"Date: {graph.date}")
print(f"Nodes: {graph.x.shape[0]}")
print(f"Edges: {graph.edge_index.shape[1]}")
print(f"Node features: {graph.x.shape[1]}")
print(f"Edge features: {graph.edge_attr.shape[1]}")

# Convert to NetworkX for visualization
G = nx.Graph()
edge_index = graph.edge_index.numpy()
for i in range(edge_index.shape[1]):
    G.add_edge(edge_index[0, i], edge_index[1, i])

# Plot
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color='lightblue', node_size=500, 
        with_labels=True, font_size=8, edge_color='gray', alpha=0.5)
plt.title(f'Graph Structure - {graph.date}')
plt.tight_layout()
plt.savefig('../plots/notebook_graph_structure.png', dpi=300, bbox_inches='tight')
plt.show()

# Graph statistics
print(f"\nGraph Statistics:")
print(f"Average degree: {sum(dict(G.degree()).values()) / len(G):.2f}")
print(f"Density: {nx.density(G):.4f}")
print(f"Is connected: {nx.is_connected(G)}")
```

## Tips & Tricks

### Magic Commands
```python
%matplotlib inline          # Display plots inline
%load_ext autoreload       # Auto-reload modules
%autoreload 2              # Reload all modules before execution
%%time                     # Time cell execution
%%timeit                   # Time repeated execution
```

### Display Options
```python
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
```

### Progress Bars
```python
from tqdm.notebook import tqdm

for i in tqdm(range(100)):
    # Your code here
    pass
```

## Version Control

### Before Committing
1. Clear all outputs: `Cell > All Output > Clear`
2. Restart kernel and run all: `Kernel > Restart & Run All`
3. Verify no errors
4. Commit

### .gitignore
Notebook checkpoints are already in `.gitignore`:
```
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/
```

## Notes

- Notebooks are for exploration, not production code
- Move stable code to `scripts/` or `models/`
- Keep notebooks focused and concise
- Document findings in markdown cells
- Export important results to `plots/` or `doc/`
