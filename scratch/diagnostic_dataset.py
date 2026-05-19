import torch
import h5py
import numpy as np
import os
import sys
import logging
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.getcwd())

from src.stgat_v2.train_resilient import ResilientSpatioTemporalDataset, STATION_REGIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnostic():
    h5_path = 'dataset_v13_train_val_M5_patched.h5'
    if not os.path.exists(h5_path):
        print(f"File not found: {h5_path}")
        return

    print(f"Opening {h5_path}...")
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"Keys: {list(f.keys())}")
            train_labels = f['train']['label_event'][:]
            print(f"Loaded {len(train_labels)} labels")
            
        event_indices = np.where(train_labels == 1.0)[0]
        print(f"Found {len(event_indices)} events")
        
        # Test dataset
        dataset = ResilientSpatioTemporalDataset(h5_path, 'train', indices=event_indices[:10])
        print(f"Dataset length: {len(dataset)}")
        
        print("Fetching first item...")
        item = dataset[0]
        print("Success fetching first item!")
        for i, val in enumerate(item):
            if isinstance(val, torch.Tensor):
                print(f"Item {i} shape: {val.shape}")
            else:
                print(f"Item {i}: {val}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnostic()
