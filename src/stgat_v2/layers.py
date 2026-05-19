import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalBatchNorm2d(nn.Module):
    """
    Implementation of Conditional Batch Normalization.
    Instead of learning static gamma/beta, it uses an external embedding (Physics Features)
    to generate the scale and shift parameters.
    """
    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Sequential(
            nn.Linear(embedding_dim, num_features * 2),
            nn.ReLU(),
            nn.Linear(num_features * 2, num_features * 2)
        )
        
        # Initialize shift to 0 and scale to 1
        self.embed[2].weight.data.zero_()
        self.embed[2].bias.data.zero_()

    def forward(self, x, embedding=None):
        # x: (B, C, H, W), embedding: (B, embedding_dim)
        if embedding is None:
            embedding = getattr(self, 'current_embedding', None)
            if embedding is None:
                raise ValueError("Embedding must be provided either as argument or via 'current_embedding' attribute.")
                
        out = self.bn(x)
        gamma_beta = self.embed(embedding) # (B, 2*C)
        gamma, beta = torch.split(gamma_beta, self.num_features, dim=1)
        
        # Reshape for broadcasting: (B, C, 1, 1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        
        return out * (1 + gamma) + beta

class PhysicsSidecarMLP(nn.Module):
    """
    MLP that maps geomagnetic indices (Kp, Dst) to a physics embedding.
    """
    def __init__(self, input_dim=2, embedding_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: (B, 2) -> (B, embedding_dim)
        return self.mlp(x)
