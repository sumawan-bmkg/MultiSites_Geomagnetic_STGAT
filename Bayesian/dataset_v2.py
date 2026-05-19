#!/usr/bin/env python3
"""
dataset_v2.py  --  STATION-SPECIFIC DYNAMIC PRIOR DATASET
=======================================================
V9.1 Implementation: Dataset yang memuat prior spesifik per stasiun.

Modifikasi dari V3_HDF5_DataLoader:
- Extract station_id dari filename metadata HDF5
- Load prior tensor yang sesuai: prior_{station_id}.pt
- Return: (x_img, x_cosmic, station_specific_prior, y_event, y_mag, y_azm)
"""

import sys
import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ── Path setup ─────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent                    # Bayesian/
_PRIORS_DIR = _THIS_DIR / "priors"                   # Bayesian/priors/


def extract_station_id_from_filename(filename: bytes) -> str:
    """
    Extract station ID from HDF5 meta filename.
    Format: event_ALR_20250824.npy -> ALR
    """
    name = filename.decode() if isinstance(filename, bytes) else filename
    if '_' in name:
        parts = name.split('_')
        if len(parts) >= 2:
            return parts[1]
    return "UNKNOWN"


class GeomagneticCosmicDatasetV2(Dataset):
    """
    ScalogramV3 Dataset V2: Station-Specific Dynamic Priors.
    Membuka HDF5 sekali di __init__ untuk mencegah I/O Bottleneck.
    """
    def __init__(self, h5_file_path, group_name='train', transform=None, 
                 priors_dir=None):
        self.h5_file_path = str(h5_file_path)
        self.group_name = group_name
        self.transform = transform
        self.priors_dir = Path(priors_dir) if priors_dir else _PRIORS_DIR
        
        # Cache untuk station-specific priors (hanya muat sekali per stasiun)
        self._prior_cache = {}
        
        try:
            with h5py.File(self.h5_file_path, 'r') as hf:
                if self.group_name not in hf:
                    raise KeyError(f"Grup {self.group_name} tidak ditemukan di HDF5")
                self.length = hf[self.group_name]['tensors'].shape[0]
                
                # Pre-extract all station IDs from metadata
                print(f"[INFO] Extracting station IDs from {self.group_name} metadata...")
                meta_data = hf[self.group_name]['meta'][:]
                self.station_ids = [
                    extract_station_id_from_filename(m) for m in meta_data
                ]
                
                # Validate station IDs against available priors
                available_priors = set()
                if self.priors_dir.exists():
                    available_priors = {p.stem.replace('prior_', '') 
                                       for p in self.priors_dir.glob('prior_*.pt')}
                
                missing = set(self.station_ids) - available_priors - {"UNKNOWN"}
                if missing:
                    print(f"[WARN] Missing priors for stations: {missing}")
                    print(f"[WARN] Will use uniform prior for missing stations.")
                
                print(f"[OK] HDF5 Registered: {self.group_name} -> {self.length} samples.")
                print(f"[OK] Stations found: {sorted(set(self.station_ids))}")
                
        except Exception as e:
            print(f"Error HDF5: {e}")
            self.length = 0
            self.station_ids = []

    def _load_station_prior(self, station_id: str) -> torch.Tensor:
        """
        Load station-specific prior tensor dari cache atau file.
        Jika tidak ditemukan, return uniform prior.
        """
        if station_id in self._prior_cache:
            return self._prior_cache[station_id]
        
        prior_path = self.priors_dir / f"prior_{station_id}.pt"
        
        if prior_path.exists():
            prior = torch.load(str(prior_path), map_location='cpu')
            self._prior_cache[station_id] = prior
            return prior
        else:
            # Fallback: uniform prior jika stasiun tidak memiliki prior
            print(f"[WARN] Prior for station {station_id} not found. Using uniform prior.")
            uniform = torch.ones(360) / 360.0
            self._prior_cache[station_id] = uniform
            return uniform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # FIX 1 (ANTI-LEAK): Buka dan tutup file secara langsung untuk tiap pemanggilan
        with h5py.File(self.h5_file_path, 'r', rdcc_nbytes=0) as hf:
            grp = hf[self.group_name]
            x_img    = np.array(grp['tensors'][idx], copy=True)
            x_cosmic = np.array(grp['cosmic_features'][idx], copy=True)
            y_event  = int(grp['label_event'][idx])
            y_mag    = float(grp['label_mag'][idx])
            y_azm    = float(grp['label_azm'][idx])
        
        # Konversi ke Tensor
        x_img    = torch.from_numpy(x_img).float()
        x_cosmic = torch.from_numpy(x_cosmic).float()
        
        # BOUNDED NORMALIZATION (Tanh) untuk mencegah exploding gradients
        kp_norm  = x_cosmic[0] / 9.0
        dst_norm = torch.tanh(x_cosmic[1] / 50.0)
        x_cosmic_safe = torch.stack([kp_norm, dst_norm]).float()
        
        # --- [V9.1] STATION-SPECIFIC DYNAMIC PRIOR ---
        station_id = self.station_ids[idx]
        station_prior = self._load_station_prior(station_id)
        
        if self.transform:
            x_img = self.transform(x_img)
            
        return x_img, x_cosmic_safe, station_prior, y_event, y_mag, y_azm
    
    def __del__(self):
        pass


def create_v91_dataloaders(h5_path, batch_size=16, num_workers=0, use_sampler=True,
                           priors_dir=None):
    """
    Create V9.1 DataLoaders with station-specific dynamic priors.
    """
    train_dataset = GeomagneticCosmicDatasetV2(h5_path, group_name='train', priors_dir=priors_dir)
    val_dataset   = GeomagneticCosmicDatasetV2(h5_path, group_name='val', priors_dir=priors_dir)
    
    sampler = None
    shuffle = True
    
    if use_sampler:
        # Hitung bobot sampel untuk menangani imbalance 1:10
        with h5py.File(h5_path, 'r') as hf:
            labels = hf['train/label_event'][:]
            
        class_counts = np.bincount(labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[labels]
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        shuffle = False
        print(f"[OK] WeightedRandomSampler Active (Class Counts: {class_counts})")

    # num_workers=0 wajib karena HDF5 tidak thread-safe
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    h5_v3 = r"d:\multi\scalogramv3\scalogram_v3_cosmic_final.h5"
    if Path(h5_v3).exists():
        ds = GeomagneticCosmicDatasetV2(h5_v3, group_name='train')
        if len(ds) > 0:
            img, cos, prior, ev, mag, azm = ds[0]
            print("\n--- Smoke Test DataLoader V9.1 ---")
            print(f"Img Shape         : {img.shape}")
            print(f"Cosmic Vector     : {cos} [Kp_norm, Dst_tanh]")
            print(f"Station Prior     : {prior.shape} (sum={prior.sum():.4f})")
            print(f"Labels            : Event={ev}, Mag={mag}, Azm={azm}")
            print("[OK] V9.1 Dataset v2 VERIFIED.")
    else:
        print("HDF5 V3 belum siap.")
