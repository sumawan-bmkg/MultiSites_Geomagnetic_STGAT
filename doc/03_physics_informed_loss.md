# Tahap 3: Physics-Informed Loss Function

**Tanggal**: 2 Mei 2026  
**Status**: Design Phase

## Tujuan

Mengintegrasikan hukum fisika atenuasi seismik ke dalam fungsi loss untuk memastikan model tidak hanya fit ke data, tetapi juga mematuhi prinsip fisika yang mendasari propagasi gelombang seismik.

## Hukum Fisika: Atenuasi Amplitudo Seismik

### Formula Dasar

Amplitudo gelombang seismik berkurang secara eksponensial dengan jarak:

```
A(d) = A₀ × e^(-α × d)
```

**Dimana**:
- `A(d)` = Amplitudo pada jarak d dari episenter
- `A₀` = Amplitudo awal di episenter
- `α` = Koefisien atenuasi (tergantung medium, frekuensi)
- `d` = Jarak dari episenter (km)

### Interpretasi Fisika

1. **Geometric Spreading**: Energi menyebar dalam volume 3D
2. **Anelastic Attenuation**: Energi diserap oleh medium
3. **Scattering**: Energi tersebar oleh heterogenitas

Koefisien α biasanya dalam range:
- **α ≈ 0.0001 - 0.001 km⁻¹** untuk kerak bumi
- Nilai lebih tinggi untuk medium yang lebih attenuative

## Multi-Task Loss Function

### Total Loss

```python
L_total = λ₁·L_event + λ₂·L_mag + λ₃·L_azm + λ₄·L_dist + λ₅·L_physics
```

**Dimana**:
- `L_event`: Binary Cross-Entropy untuk deteksi event
- `L_mag`: Mean Squared Error untuk magnitude
- `L_azm`: Angular loss untuk azimuth
- `L_dist`: Mean Squared Error untuk distance
- `L_physics`: Physics-informed loss untuk atenuasi
- `λ₁, λ₂, λ₃, λ₄, λ₅`: Hyperparameter bobot

### 1. Event Detection Loss (Classification)

```python
def event_loss(y_pred, y_true):
    """
    Binary Cross-Entropy untuk deteksi prekursor.
    
    Args:
        y_pred: [batch_size, 1] - probabilitas event
        y_true: [batch_size, 1] - ground truth (0 atau 1)
    
    Returns:
        BCE loss
    """
    return F.binary_cross_entropy_with_logits(y_pred, y_true)
```

**Handling Class Imbalance**:
- Background noise >> prekursor events
- Gunakan **weighted BCE** atau **focal loss**:

```python
def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    """
    Focal Loss untuk menangani class imbalance.
    """
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()
```

### 2. Magnitude Loss (Regression)

```python
def magnitude_loss(y_pred, y_true, mask):
    """
    MSE untuk prediksi magnitude.
    
    Args:
        y_pred: [batch_size, 1] - predicted magnitude
        y_true: [batch_size, 1] - true magnitude
        mask: [batch_size, 1] - 1 jika ada event, 0 jika background
    
    Returns:
        Masked MSE loss
    """
    # Hanya hitung loss untuk samples dengan event=1
    loss = F.mse_loss(y_pred * mask, y_true * mask, reduction='sum')
    return loss / (mask.sum() + 1e-8)
```

### 3. Azimuth Loss (Circular Regression)

Azimuth adalah variabel circular (0° = 360°), sehingga perlu loss khusus:

```python
def azimuth_loss(y_pred, y_true, mask):
    """
    Angular loss untuk azimuth prediction.
    
    Args:
        y_pred: [batch_size, 1] - predicted azimuth (degrees)
        y_true: [batch_size, 1] - true azimuth (degrees)
        mask: [batch_size, 1] - 1 jika ada event, 0 jika background
    
    Returns:
        Angular error loss
    """
    # Convert to radians
    y_pred_rad = y_pred * math.pi / 180.0
    y_true_rad = y_true * math.pi / 180.0
    
    # Circular difference
    diff = torch.atan2(
        torch.sin(y_pred_rad - y_true_rad),
        torch.cos(y_pred_rad - y_true_rad)
    )
    
    # MSE on angular difference
    loss = (diff ** 2 * mask).sum() / (mask.sum() + 1e-8)
    return loss
```

### 4. Distance Loss (Regression)

```python
def distance_loss(y_pred, y_true, mask):
    """
    MSE untuk prediksi jarak ke episenter.
    
    Args:
        y_pred: [batch_size, 24] - predicted distances untuk 24 stasiun
        y_true: [batch_size, 24] - true distances
        mask: [batch_size, 1] - 1 jika ada event, 0 jika background
    
    Returns:
        Masked MSE loss
    """
    # Expand mask untuk 24 stasiun
    mask_expanded = mask.unsqueeze(-1).expand_as(y_pred)
    
    loss = F.mse_loss(
        y_pred * mask_expanded,
        y_true * mask_expanded,
        reduction='sum'
    )
    return loss / (mask_expanded.sum() + 1e-8)
```

### 5. Physics-Informed Loss (KRITIS)

```python
def physics_loss(amplitudes_pred, distances_true, mask, alpha=0.0005):
    """
    Physics-informed loss berdasarkan hukum atenuasi seismik.
    
    Formula: A(d) = A₀ × e^(-α × d)
    
    Args:
        amplitudes_pred: [batch_size, 24] - predicted amplitudes di 24 stasiun
        distances_true: [batch_size, 24] - true distances dari episenter
        mask: [batch_size, 1] - 1 jika ada event, 0 jika background
        alpha: koefisien atenuasi (default 0.0005 km⁻¹)
    
    Returns:
        Physics compliance loss
    """
    # Expand mask
    mask_expanded = mask.unsqueeze(-1).expand_as(amplitudes_pred)
    
    # Hitung A₀ (amplitudo di episenter) dari stasiun terdekat
    # Asumsi: A₀ = A(d) × e^(α × d)
    A0_estimates = amplitudes_pred * torch.exp(alpha * distances_true)
    
    # A₀ seharusnya konsisten untuk semua stasiun
    # Variance dari A₀ estimates seharusnya kecil
    A0_mean = (A0_estimates * mask_expanded).sum(dim=1, keepdim=True) / (mask_expanded.sum(dim=1, keepdim=True) + 1e-8)
    A0_variance = ((A0_estimates - A0_mean) ** 2 * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
    
    # Alternatif: Hitung expected amplitude dan bandingkan
    # A_expected = A0_mean × e^(-α × d)
    A_expected = A0_mean * torch.exp(-alpha * distances_true)
    physics_error = F.mse_loss(
        amplitudes_pred * mask_expanded,
        A_expected * mask_expanded,
        reduction='sum'
    ) / (mask_expanded.sum() + 1e-8)
    
    # Combine variance penalty dan physics error
    return A0_variance + physics_error
```

**Interpretasi**:
- Model dipaksa untuk memprediksi amplitudo yang konsisten dengan hukum atenuasi
- Jika model memprediksi amplitudo yang tidak realistis, physics loss akan tinggi

## Hyperparameter Tuning

### Loss Weights (λ)

**Initial Values** (starting point):
```python
lambda_event = 1.0      # Event detection (paling penting)
lambda_mag = 0.5        # Magnitude prediction
lambda_azm = 0.3        # Azimuth prediction
lambda_dist = 0.5       # Distance prediction
lambda_physics = 0.2    # Physics compliance
```

**Tuning Strategy**:
1. Train dengan λ_physics = 0 (baseline tanpa physics)
2. Gradually increase λ_physics: 0.1 → 0.2 → 0.5
3. Monitor trade-off antara data fit dan physics compliance

### Attenuation Coefficient (α)

**Learnable Parameter** (Recommended):
```python
class PhysicsInformedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ... other layers
        
        # Learnable attenuation coefficient
        self.alpha = nn.Parameter(torch.tensor(0.0005))
        
    def forward(self, x):
        # ... forward pass
        
        # Use self.alpha in physics loss
        physics_loss = self.compute_physics_loss(amplitudes, distances, self.alpha)
        return outputs, physics_loss
```

**Constraint**: α harus positif
```python
# Clamp alpha to positive values
self.alpha.data = torch.clamp(self.alpha.data, min=1e-6, max=0.01)
```

## Implementation Example

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_event=1.0, lambda_mag=0.5, lambda_azm=0.3,
                 lambda_dist=0.5, lambda_physics=0.2):
        super().__init__()
        self.lambda_event = lambda_event
        self.lambda_mag = lambda_mag
        self.lambda_azm = lambda_azm
        self.lambda_dist = lambda_dist
        self.lambda_physics = lambda_physics
        
        # Learnable attenuation coefficient
        self.alpha = nn.Parameter(torch.tensor(0.0005))
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with keys ['event', 'mag', 'azm', 'dist', 'amplitudes']
            targets: dict with keys ['event', 'mag', 'azm', 'dist']
        
        Returns:
            total_loss, loss_dict
        """
        # Event mask (1 jika ada event, 0 jika background)
        event_mask = targets['event']
        
        # Individual losses
        l_event = self.event_loss(predictions['event'], targets['event'])
        l_mag = self.magnitude_loss(predictions['mag'], targets['mag'], event_mask)
        l_azm = self.azimuth_loss(predictions['azm'], targets['azm'], event_mask)
        l_dist = self.distance_loss(predictions['dist'], targets['dist'], event_mask)
        l_physics = self.physics_loss(
            predictions['amplitudes'],
            targets['dist'],
            event_mask,
            self.alpha
        )
        
        # Total loss
        total_loss = (
            self.lambda_event * l_event +
            self.lambda_mag * l_mag +
            self.lambda_azm * l_azm +
            self.lambda_dist * l_dist +
            self.lambda_physics * l_physics
        )
        
        # Return loss breakdown for logging
        loss_dict = {
            'total': total_loss.item(),
            'event': l_event.item(),
            'magnitude': l_mag.item(),
            'azimuth': l_azm.item(),
            'distance': l_dist.item(),
            'physics': l_physics.item(),
            'alpha': self.alpha.item()
        }
        
        return total_loss, loss_dict
```

## Validation Metrics

### Physics Compliance Check

Setelah training, validasi apakah model mematuhi hukum fisika:

```python
def validate_physics_compliance(model, test_loader):
    """
    Hitung seberapa baik prediksi model mematuhi hukum atenuasi.
    """
    alpha_learned = model.loss_fn.alpha.item()
    
    physics_errors = []
    for batch in test_loader:
        with torch.no_grad():
            predictions = model(batch)
            
            # Hitung expected amplitude
            A0 = predictions['amplitudes'].max(dim=1, keepdim=True)[0]
            A_expected = A0 * torch.exp(-alpha_learned * batch.y_dist)
            
            # Error
            error = torch.abs(predictions['amplitudes'] - A_expected)
            physics_errors.append(error.mean().item())
    
    mean_physics_error = np.mean(physics_errors)
    print(f"Mean Physics Compliance Error: {mean_physics_error:.4f}")
    print(f"Learned Attenuation Coefficient α: {alpha_learned:.6f} km⁻¹")
    
    return mean_physics_error, alpha_learned
```

## Expected Results

### Learned α Value
- **Expected range**: 0.0001 - 0.001 km⁻¹
- Jika α terlalu besar (> 0.01): Model overfitting ke noise
- Jika α terlalu kecil (< 0.00001): Model tidak belajar atenuasi

### Physics Loss Trend
- **Epoch 1-10**: Physics loss tinggi (model belum belajar fisika)
- **Epoch 10-50**: Physics loss turun drastis
- **Epoch 50+**: Physics loss stabil (model sudah comply dengan fisika)

## Next Steps

Setelah physics loss design selesai:
1. Implement loss functions dalam PyTorch
2. Test dengan synthetic data
3. Lanjut ke **Tahap 4: Chronological Split Implementation**

## Changelog

- **2026-05-02**: Dokumentasi physics loss design dibuat
- **Pending**: Implementasi dan testing
