# Plots Directory

This directory contains visualizations and plots generated during analysis and evaluation.

## Plot Categories

### 1. Data Exploration
- Station distribution map
- Earthquake epicenter map
- Magnitude distribution histogram
- Temporal distribution timeline
- Tectonic region visualization

### 2. Graph Visualization
- Graph snapshot examples
- Adjacency matrix heatmap
- Edge weight distribution
- Node feature distributions

### 3. Training Progress
- Loss curves (train vs val)
- Metric curves (accuracy, F1, etc.)
- Learning rate schedule
- Physics loss evolution
- Attenuation coefficient (α) evolution

### 4. Model Evaluation
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Magnitude prediction scatter plot
- Distance prediction error map
- Azimuth prediction polar plot

### 5. Physics Compliance
- Attenuation curve fit
- Amplitude vs distance scatter
- Physics error distribution
- Learned α comparison with literature

### 6. Error Analysis
- False positive/negative examples
- Magnitude error by range
- Distance error by station
- Temporal error patterns

## File Naming Convention

```
{category}_{description}_{date}.{ext}

Examples:
- data_station_map_20260502.png
- training_loss_curves_20260502.png
- eval_confusion_matrix_20260502.png
- physics_attenuation_fit_20260502.png
```

## Recommended Tools

### Python Libraries
```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
```

### Example: Station Map
```python
import matplotlib.pyplot as plt
import pandas as pd

# Load station data
stations = pd.read_csv('data/processed/lokasi_stasiun_clean.csv')

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(stations['Longitude'], stations['Latitude'], 
           c=stations['region_idx'], cmap='viridis', s=100)
plt.colorbar(label='Region Index')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('24 Seismic Stations in Indonesia')
plt.grid(True, alpha=0.3)
plt.savefig('plots/data_station_map.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Example: Loss Curves
```python
import matplotlib.pyplot as plt
import json

# Load training log
with open('models/logs/training_log.json', 'r') as f:
    log = json.load(f)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Total loss
axes[0, 0].plot(log['train_loss'], label='Train')
axes[0, 0].plot(log['val_loss'], label='Validation')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Total Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Event loss
axes[0, 1].plot(log['train_event_loss'], label='Train')
axes[0, 1].plot(log['val_event_loss'], label='Validation')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Event Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Magnitude loss
axes[1, 0].plot(log['train_mag_loss'], label='Train')
axes[1, 0].plot(log['val_mag_loss'], label='Validation')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Magnitude Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Physics loss
axes[1, 1].plot(log['train_physics_loss'], label='Train')
axes[1, 1].plot(log['val_physics_loss'], label='Validation')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Physics Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/training_loss_curves.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Interactive Plots

For interactive exploration, use Plotly:

```python
import plotly.graph_objects as go

# Create interactive station map
fig = go.Figure(data=go.Scattergeo(
    lon=stations['Longitude'],
    lat=stations['Latitude'],
    text=stations['Kode Stasiun'],
    mode='markers+text',
    marker=dict(size=10, color=stations['region_idx'], colorscale='Viridis')
))

fig.update_layout(
    title='Interactive Station Map',
    geo=dict(
        scope='asia',
        center=dict(lat=-2, lon=118),
        projection_scale=3
    )
)

fig.write_html('plots/interactive_station_map.html')
```

## Notes

- Large plot files (PNG, PDF) are in `.gitignore`
- Keep example plots for documentation
- Use high DPI (300) for publication-quality plots
- Save both PNG (for quick view) and PDF (for papers)
