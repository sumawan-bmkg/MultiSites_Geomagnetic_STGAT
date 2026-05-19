#!/usr/bin/env python3
"""
dataset_v3.py  --  STATION EMBEDDING AWARE DATASET
=================================================
V9.4 Implementation: Menambahkan station_id sebagai integer untuk embedding.
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ── Path setup ─────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent                    # Bayesian/
_PRIORS_DIR = _THIS_DIR / "priors"                   # Bayesian/priors/

STATION_MAP = {
    'ALR': 1, 'AMB': 2, 'CLP': 3, 'GTO': 4, 'KPY': 5, 
    'LPS': 6, 'LUT': 7, 'LWA': 8, 'LWK': 9, 'MLB': 10, 
    'PLU': 11, 'ROT': 12, 'SBG': 13, 'SCN': 14, 'SKB': 15, 
    'SMI': 16, 'SRG': 17, 'SRO': 18, 'TNT': 19, 'TRD': 20, 
    'TRT': 21, 'YOG': 22
}

def extract_station_id_from_filename(filename: bytes) -> str:
    name = filename.decode() if isinstance(filename, bytes) else filename
    if '_' in name:
        parts = name.split('_')
        if len(parts) >= 2:
            return parts[1]
    return "UNKNOWN"

class GeomagneticCosmicDatasetV3(Dataset):
    def __init__(self, h5_file_path, group_name='train', transform=None, 
                 priors_dir=None):
        self.h5_file_path = str(h5_file_path)
        self.group_name = group_name
        self.transform = transform
        self.priors_dir = Path(priors_dir) if priors_dir else _PRIORS_DIR
        self._prior_cache = {}
        
        with h5py.File(self.h5_file_path, 'r') as hf:
            self.length = hf[self.group_name]['tensors'].shape[0]
            meta_data = hf[self.group_name]['meta'][:]
            self.station_codes = [extract_station_id_from_filename(m) for m in meta_data]
            self.station_ids = [STATION_MAP.get(c, 0) for c in self.station_codes]
            print(f"[OK] HDF5 Registered: {self.group_name} -> {self.length} samples.")

    def _load_station_prior(self, station_id_str: str) -> torch.Tensor:
        if station_id_str in self._prior_cache:
            return self._prior_cache[station_id_str]
        prior_path = self.priors_dir / f"prior_{station_id_str}.pt"
        if prior_path.exists():
            prior = torch.load(str(prior_path), map_location='cpu')
            self._prior_cache[station_id_str] = prior
            return prior
        else:
            uniform = torch.ones(360) / 360.0
            self._prior_cache[station_id_str] = uniform
            return uniform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r', rdcc_nbytes=0) as hf:
            grp = hf[self.group_name]
            x_img    = np.array(grp['tensors'][idx], copy=True)
            x_cosmic = np.array(grp['cosmic_features'][idx], copy=True)
            y_event  = int(grp['label_event'][idx])
            y_mag    = float(grp['label_mag'][idx])
            y_azm    = float(grp['label_azm'][idx])
        
        x_img    = torch.from_numpy(x_img).float()
        x_cosmic = torch.from_numpy(x_cosmic).float()
        kp_norm  = x_cosmic[0] / 9.0
        dst_norm = torch.tanh(x_cosmic[1] / 50.0)
        x_cosmic_safe = torch.stack([kp_norm, dst_norm]).float()
        
        stn_code = self.station_codes[idx]
        stn_id   = self.station_ids[idx]
        station_prior = self._load_station_prior(stn_code)
        
        if self.transform:
            x_img = self.transform(x_img)
            
        return x_img, x_cosmic_safe, station_prior, torch.tensor(stn_id, dtype=torch.long), y_event, y_mag, y_azm

def create_v94_dataloaders(h5_path, batch_size=16, num_workers=0, use_sampler=True,
                           priors_dir=None):
    train_loader = None
    with h5py.File(h5_path, 'r') as hf:
        if 'train' in hf:
            train_dataset = GeomagneticCosmicDatasetV3(h5_path, group_name='train', priors_dir=priors_dir)
            sampler = None
            shuffle = True
            if use_sampler:
                labels = hf['train/label_event'][:]
                class_counts = np.bincount(labels)
                class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
                sample_weights = class_weights[labels]
                sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
                shuffle = False
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers)
    
    val_dataset   = GeomagneticCosmicDatasetV3(h5_path, group_name='val', priors_dir=priors_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
