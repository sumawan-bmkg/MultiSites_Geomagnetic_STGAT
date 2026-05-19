import torch
import torch.nn as nn
from torchvision import models
import gc

def check_mem():
    device = 'cpu'
    print(f"Device: {device}")
    
    model = models.efficientnet_b1(weights=None).features.to(device)
    for p in model.parameters(): p.requires_grad = True
    
    # Try 1 station only
    print("Trying 1 station...")
    x = torch.randn(1, 3, 128, 1440).to(device)
    x.requires_grad = True
    
    from torch.utils.checkpoint import checkpoint
    out = checkpoint(model, x, use_reentrant=False)
    print(f"Forward done. Shape: {out.shape}")
    
    out.sum().backward()
    print("Backward done.")

if __name__ == "__main__":
    check_mem()
