# Tahap 5: Model Architecture

**Tanggal**: 2 Mei 2026  
**Status**: Design Phase

## Overview

Model ANTIGRAVITY menggabungkan dua komponen utama:
1. **Graph Attention Network (GAT)** - untuk spatial feature learning
2. **Dynamic Physics-Informed Neural Network (DPINN)** - untuk physics compliance

## Architecture Diagram

```
Input Graph (24 nodes, edges)
         |
         v
    [GAT Layer 1]  (128 channels, 4 heads)
         |
         v
    [GAT Layer 2]  (128 channels, 4 heads)
         |
         v
    [GAT Layer 3]  (128 channels, 4 heads)
         |
         v
    [Global Pooling]
         |
         +---> [Event Head] -----> Binary Classification (event/background)
         |
         +---> [Magnitude Head] --> Regression (Mw)
         |
         +---> [Azimuth Head] ----> Circular Regression (degrees)
         |
         +---> [Distance Head] ---> Multi-output Regression (24 distances)
         |
         v
    [Physics Loss]
    (Attenuation: A = A₀ × e^(-α×d))
```

## Component Details

### 1. Graph Attention Network (GAT)

#### Why GAT?
- **Attention Mechanism**: Automatically learns importance of each neighbor
- **Multi-head Attention**: Captures different types of relationships
- **Permutation Invariant**: Order of nodes doesn't matter
- **Handles Variable Connectivity**: Works with missing nodes

#### Architecture

```python
class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )
        
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, 
                       heads=heads, dropout=dropout)
            )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
```

#### Attention Mechanism

Untuk setiap node i dan neighbor j:

```
α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
h'_i = σ(Σ_j α_ij W h_j)
```

Dimana:
- `α_ij` = attention coefficient dari node j ke node i
- `W` = learnable weight matrix
- `a` = learnable attention vector
- `||` = concatenation
- `σ` = activation function (ELU)

### 2. Multi-Task Output Heads

#### Event Detection Head

```python
class EventHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x: [batch_size, in_channels] (after global pooling)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Logits (no sigmoid, use BCEWithLogitsLoss)
        return x
```

#### Magnitude Head

```python
class MagnitudeHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Output: magnitude (typically 5.0 - 8.0)
        return x
```

#### Azimuth Head (Circular Regression)

```python
class AzimuthHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        # Output sin and cos components for circular regression
        self.fc_sin = nn.Linear(hidden_channels, 1)
        self.fc_cos = nn.Linear(hidden_channels, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        sin_component = self.fc_sin(x)
        cos_component = self.fc_cos(x)
        
        # Convert to azimuth (0-360 degrees)
        azimuth = torch.atan2(sin_component, cos_component)
        azimuth = torch.rad2deg(azimuth)
        azimuth = (azimuth + 360) % 360
        
        return azimuth
```

**Why Sin/Cos Representation?**
- Azimuth is circular: 0° = 360°
- Direct regression would treat 1° and 359° as very different
- Sin/Cos representation preserves circular nature

#### Distance Head (Multi-output)

```python
class DistanceHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_stations=24):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_stations)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        distances = self.fc2(x)
        # Output: [batch_size, 24] distances in km
        # Apply ReLU to ensure positive distances
        distances = F.relu(distances)
        return distances
```

### 3. Complete Model

```python
class ANTIGRAVITYModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # GAT Encoder
        self.encoder = GATEncoder(
            in_channels=config['model']['input_features'],
            hidden_channels=config['model']['gat']['hidden_channels'],
            num_layers=config['model']['gat']['num_layers'],
            heads=config['model']['gat']['heads'],
            dropout=config['model']['gat']['dropout']
        )
        
        # Global pooling
        self.pool = global_mean_pool
        
        # Output heads
        encoder_out_channels = (config['model']['gat']['hidden_channels'] * 
                               config['model']['gat']['heads'])
        
        self.event_head = EventHead(encoder_out_channels, 
                                    config['model']['output']['event_hidden'])
        self.magnitude_head = MagnitudeHead(encoder_out_channels,
                                           config['model']['output']['magnitude_hidden'])
        self.azimuth_head = AzimuthHead(encoder_out_channels,
                                       config['model']['output']['azimuth_hidden'])
        self.distance_head = DistanceHead(encoder_out_channels,
                                         config['model']['output']['distance_hidden'])
        
        # Learnable attenuation coefficient (for physics loss)
        self.alpha = nn.Parameter(
            torch.tensor(config['model']['dpinn']['alpha_init'])
        )
    
    def forward(self, data):
        # Encode graph
        x = self.encoder(data.x, data.edge_index, data.edge_attr)
        
        # Global pooling (graph-level representation)
        x_global = self.pool(x, data.batch)
        
        # Multi-task predictions
        event_logits = self.event_head(x_global)
        magnitude = self.magnitude_head(x_global)
        azimuth = self.azimuth_head(x_global)
        distances = self.distance_head(x_global)
        
        # For physics loss, we need node-level amplitudes
        # (This would come from actual seismic features in real implementation)
        amplitudes = x.mean(dim=1)  # Placeholder
        
        return {
            'event': event_logits,
            'magnitude': magnitude,
            'azimuth': azimuth,
            'distances': distances,
            'amplitudes': amplitudes
        }
    
    def get_alpha(self):
        """Get current attenuation coefficient (clamped to valid range)."""
        return torch.clamp(self.alpha, min=1e-6, max=0.01)
```

## Global Pooling Strategy

### Options

1. **Global Mean Pool** (Default)
   ```python
   x_global = global_mean_pool(x, batch)
   ```
   - Simple and effective
   - Treats all nodes equally

2. **Global Max Pool**
   ```python
   x_global = global_max_pool(x, batch)
   ```
   - Captures strongest signals
   - May miss distributed patterns

3. **Global Attention Pool**
   ```python
   x_global = global_attention_pool(x, batch)
   ```
   - Learns which nodes are important
   - More parameters to train

**Recommendation**: Start with Global Mean Pool, experiment with Attention Pool if needed.

## Input Features

### Node Features (per station)

Current implementation (basic):
```python
node_features = [
    latitude,      # Normalized
    longitude,     # Normalized
    elevation,     # Normalized
    region_idx,    # 0, 1, or 2
    is_active      # 1 or 0
]
```

**Future enhancements** (when seismic data available):
```python
node_features = [
    # Spatial
    latitude, longitude, elevation, region_idx,
    
    # Seismic signals
    amplitude_mean, amplitude_std,
    frequency_dominant, frequency_bandwidth,
    signal_to_noise_ratio,
    
    # Temporal features
    signal_trend,  # Increasing/decreasing
    anomaly_score,
    
    # Status
    is_active
]
```

### Edge Features

```python
edge_features = [
    distance,          # Haversine distance (km)
    tectonic_penalty,  # 0.0 or 0.5
    weight            # 1 / (distance + ε)
]
```

## Model Size & Complexity

### Parameter Count

With default configuration:
- GAT Encoder: ~500K parameters
- Output Heads: ~50K parameters
- **Total**: ~550K parameters

### Memory Requirements

- Training batch (32 graphs): ~200 MB GPU memory
- Model: ~2 MB
- **Total**: ~500 MB GPU memory (comfortable for most GPUs)

### Inference Speed

- Single graph: ~5 ms (on GPU)
- Batch of 32: ~20 ms
- **Throughput**: ~1,600 graphs/second

## Training Strategy

### Phase 1: Baseline (No Physics)
1. Train with λ_physics = 0
2. Focus on event detection and magnitude prediction
3. Establish baseline performance

### Phase 2: Physics Integration
1. Gradually increase λ_physics: 0.1 → 0.2 → 0.5
2. Monitor physics compliance metrics
3. Ensure α converges to realistic value

### Phase 3: Fine-tuning
1. Adjust loss weights based on validation performance
2. Experiment with different GAT architectures
3. Add Cosmic Gating module if needed

## Validation Metrics

### Event Detection
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

### Magnitude Prediction
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² Score

### Azimuth Prediction
- Angular Error (degrees)
- Circular Correlation

### Distance Prediction
- MAE per station
- RMSE per station

### Physics Compliance
- Mean Physics Error
- Learned α value
- Attenuation curve fit

## Next Steps

1. Implement model in PyTorch
2. Create training script
3. Test with synthetic data
4. Train on real data
5. Evaluate on blind test set

## Changelog

- **2026-05-02**: Initial architecture design
- **Pending**: Implementation and testing
