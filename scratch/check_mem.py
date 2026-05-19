import torch
import torch.nn as nn
from torchvision import models
import gc

def check_mem():
    device = 'cpu'
    print(f"Device: {device}")
    
    # EfficientNet-B1 features
    model = models.efficientnet_b1(weights=None).features.to(device)
    for p in model.parameters(): p.requires_grad = True
    
    # Mock input: (24, 3, 128, 1440)
    x = torch.randn(24, 3, 128, 1440).to(device)
    x.requires_grad = True
    
    print("Forward pass starting...")
    # Option 1: Standard
    # out = model(x)
    
    # Option 2: Checkpointed
    from torch.utils.checkpoint import checkpoint
    out = checkpoint(model, x, use_reentrant=False)
    
    print(f"Forward done. Output shape: {out.shape}")
    
    loss = out.sum()
    print("Backward pass starting...")
    loss.backward()
    print("Backward done.")

if __name__ == "__main__":
    try:
        check_mem()
    except Exception as e:
        print(f"Error: {e}")
