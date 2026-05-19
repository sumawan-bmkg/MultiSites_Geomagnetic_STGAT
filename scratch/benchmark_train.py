import torch
import time
import os
import sys
import logging
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.getcwd())

from src.stgat_v2.train_resilient import ResilientSpatioTemporalDataset, ScalogramV4_ResilientSTGAT, PhysicsResilientLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark():
    device = torch.device('cpu')
    h5_path = 'dataset_v13_train_val_M5_patched.h5'
    
    print("Initializing model...")
    model = ScalogramV4_ResilientSTGAT(n_regions=6, use_cbn=True, is_autoencoder=True).to(device)
    criterion = PhysicsResilientLoss(lambda_recon=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("Loading one sample...")
    dataset = ResilientSpatioTemporalDataset(h5_path, 'train', indices=[0])
    loader = DataLoader(dataset, batch_size=1)
    
    batch = next(iter(loader))
    x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx = [b.to(device) for b in batch]
    
    print(f"Input shape: {x_img.shape}")
    
    print("Starting forward pass benchmark...")
    start_time = time.time()
    preds = model(x_img, x_cosmic)
    forward_time = time.time() - start_time
    print(f"Forward pass took: {forward_time:.2f} seconds")
    
    print("Starting backward pass benchmark...")
    targets = (y_event, y_mag, y_azm, y_dist, region_idx, x_img)
    loss, metrics = criterion(preds, targets)
    
    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time
    print(f"Backward pass took: {backward_time:.2f} seconds")
    
    print(f"Total iteration time: {forward_time + backward_time:.2f} seconds")

if __name__ == "__main__":
    benchmark()
