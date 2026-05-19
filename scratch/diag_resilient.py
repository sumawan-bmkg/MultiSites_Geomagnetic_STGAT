import torch
import h5py
from src.stgat_v2.model import ScalogramV4_ResilientSTGAT
from src.stgat_v2.train_resilient import ResilientSpatioTemporalDataset

import time

def diag():
    device = torch.device('cpu')
    print("Initializing model...")
    t0 = time.time()
    model = ScalogramV4_ResilientSTGAT(n_regions=6, use_cbn=True, is_autoencoder=True).to(device)
    print(f"Model init took: {time.time()-t0:.2f}s")
    
    print("Initializing dataset...")
    t0 = time.time()
    ds = ResilientSpatioTemporalDataset('dataset_v13_train_val_M5_patched.h5', 'train')
    print(f"Dataset init took: {time.time()-t0:.2f}s")
    
    print(f"Dataset size: {len(ds)}")
    t0 = time.time()
    x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx = ds[0]
    print(f"Data loading took: {time.time()-t0:.2f}s")
    print(f"Data loaded: img={x_img.shape}, cosmic={x_cosmic.shape}")
    
    # Test with all 24 stations
    x_img = x_img[:24].unsqueeze(0).to(device) # (B=1, N=24, 3, 128, 1440)
    x_cosmic = x_cosmic.unsqueeze(0).to(device) # (B=1, 2)
    print(f"Testing shapes: img={x_img.shape}, cosmic={x_cosmic.shape}")
    
    print("Running forward pass...")
    t0 = time.time()
    try:
        preds = model(x_img, x_cosmic)
        print(f"Forward pass success! Took: {time.time()-t0:.2f}s")
        for i, p in enumerate(preds):
            if p is not None:
                print(f"Pred {i} shape: {p.shape}")
    except Exception as e:
        print(f"Forward pass FAILED after {time.time()-t0:.2f}s: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diag()
