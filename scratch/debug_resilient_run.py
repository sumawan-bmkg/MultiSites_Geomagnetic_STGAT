import torch
import logging
import faulthandler
faulthandler.enable()
from src.stgat_v2.train_resilient import ResilientSpatioTemporalDataset, STATION_REGIONS
from src.stgat_v2.model import ScalogramV4_ResilientSTGAT
from src.stgat_v2.losses import PhysicsResilientLoss
from torch.utils.data import DataLoader
import numpy as np
import h5py
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_run():
    device = torch.device('cpu')
    h5_path = 'dataset_v13_train_val_M5_patched.h5'
    
    logger.info("Loading dataset...")
    with h5py.File(h5_path, 'r') as f:
        train_labels = f['train']['label_event'][:]
    
    event_idx = np.where(train_labels == 1.0)[0]
    noise_idx = np.where(train_labels == 0.0)[0]
    
    np.random.seed(42)
    sampled_noise_idx = np.random.choice(noise_idx, size=min(10, len(noise_idx)), replace=False)
    train_indices = np.concatenate([event_idx[:10], sampled_noise_idx])
    
    dataset = ResilientSpatioTemporalDataset(h5_path, 'train', indices=train_indices)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    logger.info(f"Loader length: {len(loader)}")
    
    model = ScalogramV4_ResilientSTGAT(n_regions=6, use_cbn=True, is_autoencoder=True).to(device)
    criterion = PhysicsResilientLoss(lambda_recon=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    try:
        for i, batch in enumerate(loader):
            logger.info(f"Iteration {i} started")
            x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            preds = model(x_img, x_cosmic)
            
            targets = (y_event, y_mag, y_azm, y_dist, region_idx, x_img)
            loss, metrics = criterion(preds, targets)
            
            logger.info(f"Iteration {i} loss: {loss.item()}")
            
            loss.backward()
            optimizer.step()
            
            if i >= 1: break
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e
    
    logger.info("Debug run finished successfully")

if __name__ == "__main__":
    debug_run()
