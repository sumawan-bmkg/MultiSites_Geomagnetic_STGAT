import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import sys
import logging
from tqdm import tqdm
import gc
import time

# Allow absolute imports when running as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from model import ScalogramV4_ResilientSTGAT
from losses import PhysicsResilientLoss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATION_REGIONS = {
    'SBG': 0, 'SCN': 0, 'KPY': 0, 'GSI': 0, 'MLB': 0,
    'LWA': 1, 'LPS': 1, 'SRG': 1, 'SKB': 1, 'CLP': 1, 'YOG': 1, 'TRT': 1,
    'LUT': 2, 'ALR': 2,
    'TNT': 3, 'TND': 3, 'GTO': 3, 'LWK': 3, 'PLU': 3, 'AMB': 3, 'SMI': 3,
    'SRO': 4, 'JYP': 4,
    'TRD': 5
}

class ResilientSpatioTemporalDataset(Dataset):
    def __init__(self, h5_path, split='train', indices=None):
        self.h5_path = h5_path
        self.split = split
        self.indices = indices
        self.h5_file = None
        self._stations = None
        self._station_to_region = None
        self.n_samples = len(self.indices) if self.indices is not None else None

    @property
    def stations(self):
        if self._stations is None:
            with h5py.File(self.h5_path, 'r') as f:
                self._stations = [s.decode() if isinstance(s, bytes) else s for s in f[self.split].attrs['stations']]
        return self._stations

    @property
    def station_to_region(self):
        if self._station_to_region is None:
            self._station_to_region = [STATION_REGIONS.get(s, 0) for s in self.stations]
        return self._station_to_region

    def __len__(self):
        if self.n_samples is not None:
            return self.n_samples
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        return len(self.h5_file[self.split]['dates'])

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        
        g = self.h5_file[self.split]
        real_idx = self.indices[idx] if self.indices is not None else idx
        
        x_img = torch.tensor(g['tensors'][real_idx]).float()
        x_cosmic = torch.tensor(g['cosmic_features'][real_idx]).float()
        y_event = torch.tensor(g['label_event'][real_idx], dtype=torch.float32)
        y_mag = torch.tensor(g['label_mag'][real_idx], dtype=torch.float32)
        label_dist = g['label_dist'][real_idx]
        y_dist = torch.tensor(label_dist).float()

        if y_event == 1.0:
            azms = g['label_azm'][real_idx]
            y_azm = torch.tensor(np.median(azms[azms > 0])).float() if np.any(azms > 0) else torch.tensor(0.0).float()
            nearest_idx = np.argmin(label_dist)
            region_idx = self.station_to_region[nearest_idx]
        else:
            y_azm = torch.tensor(0.0).float()
            region_idx = 0

        return x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx

def train_resilient_stgat():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training Resilient STGAT on {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    h5_path = os.path.join(project_dir, 'dataset_v13_train_val_M5_patched.h5')
    if not os.path.exists(h5_path):
        h5_path = 'dataset_v13_train_val_M5_patched.h5'
    
    logger.info("Performing Smart Subsampling to rescue RAM...")
    with h5py.File(h5_path, 'r') as f:
        train_labels = f['train']['label_event'][:]
        
    event_indices = np.where(train_labels == 1.0)[0]
    noise_indices = np.where(train_labels == 0.0)[0]
    
    # KEMBALI KE SMART SUBSAMPLING
    print("Menerapkan Smart Subsampling (100% Event, 25% Noise)...")
    # Ambil 25% noise
    np.random.seed(42)
    sampled_noise_indices = np.random.choice(noise_indices, size=int(len(noise_indices) * 0.25), replace=False)
    
    final_indices = np.concatenate([event_indices, sampled_noise_indices])
    np.random.shuffle(final_indices)
    print(f"Total sampel training: {len(final_indices)}")
    
    train_indices = final_indices
    
    train_dataset = ResilientSpatioTemporalDataset(h5_path, 'train', indices=train_indices)
    
    # Subsample validation set for faster feedback (limit to 2 for fast micro-batch sanity check)
    with h5py.File(h5_path, 'r') as f:
        val_size = len(f['val']['label_event'])
    # Subsample validation set to 20 samples for rapid evaluation
    np.random.seed(42)
    val_indices = np.random.choice(np.arange(val_size), size=20, replace=False)
    val_dataset = ResilientSpatioTemporalDataset(h5_path, 'val', indices=val_indices) 
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)
    
    model = ScalogramV4_ResilientSTGAT(n_regions=6, use_cbn=True, is_autoencoder=False, chunk_size=1).to(device)
    
    print("Menerapkan Absolute Freeze pada spatial_encoder...")
    for param in model.spatial_encoder.parameters():
        param.requires_grad = False
    
    v3_ckpt = 'd:/multi/scalogramv3/checkpoints/v3_v8_best.pth'
    latest_ckpt_path = os.path.join(project_dir, 'src', 'stgat_v2', 'checkpoints', 'stgat_resilient_latest.pth')
    
    start_epoch_warmup = 0
    start_epoch_fine = 0
    phase = 1
    best_val_loss = float('inf')
    
    if os.path.exists(latest_ckpt_path):
        logger.info(f"Auto-resuming from {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        phase = ckpt.get('phase', 1)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        if phase == 1:
            start_epoch_warmup = ckpt.get('epoch', 0) + 1
        else:
            start_epoch_fine = ckpt.get('epoch', 0) + 1
            start_epoch_warmup = 10 # Skip warm-up loop
    elif os.path.exists(v3_ckpt):
        logger.info(f"Loading base weights from {v3_ckpt}")
        v3_state = torch.load(v3_ckpt, map_location=device, weights_only=False)['model_state_dict']
        model.load_state_dict(v3_state, strict=False)
    
    criterion = PhysicsResilientLoss(lambda1=1.0, lambda2=2.0, lambda3=0.01, lambda4=0.1, lambda_recon=0.0).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # Fase 1: Warm-up
    print("\n" + "="*50)
    print("OOM PROTECTION: Freezing spatial_encoder (EfficientNet)...")
    print("Strategy: Sequential Station processing enabled in model.py")
    print("Strategy: Reconstruction Loss bypassed (lambda_recon=0.0)")
    print("="*50 + "\n")
    
    for param in model.spatial_encoder.parameters():
        param.requires_grad = False
    
    epochs_warmup = 3
    
    for epoch in range(start_epoch_warmup, epochs_warmup):
        if phase > 1: break
        model.train()
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        
        train_metrics = []
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs_warmup} [Warm-up]", leave=False)
        for i, batch in enumerate(loop):
            x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            preds = model(x_img, x_cosmic)
            targets = (y_event, y_mag, y_azm, y_dist, region_idx, x_img)
            loss, metrics = criterion(preds, targets)
            
            if torch.isnan(loss):
                logger.warning(f"NaN loss at sample {i}, skipping...")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_metrics.append(metrics['total'])
            loop.set_postfix(loss=loss.item())
            
            if (i + 1) % 5 == 0:
                logger.info(f"Iter {i+1}/{len(train_loader)} - Loss: {metrics['total']:.2f}, BCE: {metrics['bce']:.4f}")
                sys.stdout.flush()
            
            del x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx, preds, loss, metrics, batch
            if (i + 1) % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        gc.collect()
            
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx = [b.to(device) for b in batch]
                preds = model(x_img, x_cosmic)
                targets = (y_event, y_mag, y_azm, y_dist, region_idx, x_img)
                loss, _ = criterion(preds, targets)
                val_losses.append(loss.item())
                del x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx, preds, loss, batch
        
        avg_val_loss = np.mean(val_losses)
        logger.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'phase': 1,
            'model_state_dict': model.state_dict(),
            'best_val_loss': best_val_loss
        }, latest_ckpt_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(project_dir, 'src', 'stgat_v2', 'checkpoints', 'stgat_resilient_warmup_best.pth'))
            logger.info("Saved best warm-up.")

    # Fase 2: Global Fine-tuning (Keep spatial_encoder frozen)
    logger.info("Starting Fase 2: Global Fine-tuning...")
    for name, param in model.named_parameters():
        if "spatial_encoder" not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    epochs_fine = 3
    
    for epoch in range(start_epoch_fine, epochs_fine):
        model.train()
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
                
        train_metrics = []
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs_fine} [Fine-tune]", leave=False)
        for i, batch in enumerate(loop):
            try:
                x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx = [b.to(device) for b in batch]
                optimizer.zero_grad()
                preds = model(x_img, x_cosmic)
                targets = (y_event, y_mag, y_azm, y_dist, region_idx, x_img)
                loss, metrics = criterion(preds, targets)
                
                if torch.isnan(loss): continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_metrics.append(metrics['total'])
                loop.set_postfix(loss=loss.item())
                if (i + 1) % 5 == 0:
                    logger.info(f"Fine-tune Iter {i+1}/{len(train_loader)} - Loss: {metrics['total']:.2f}")
                    sys.stdout.flush()
                
                del x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx, preds, loss, metrics, batch
                if (i + 1) % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            except Exception as e:
                logger.error(f"Error: {e}")
        
        gc.collect()
        # Validation (Same as above)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx = [b.to(device) for b in batch]
                preds = model(x_img, x_cosmic)
                targets = (y_event, y_mag, y_azm, y_dist, region_idx, x_img)
                loss, _ = criterion(preds, targets)
                val_losses.append(loss.item())
                del x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, region_idx, preds, loss, batch
        
        avg_val_loss = np.mean(val_losses)
        logger.info(f"Fine-tune Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'phase': 2,
            'model_state_dict': model.state_dict(),
            'best_val_loss': best_val_loss
        }, latest_ckpt_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(project_dir, 'src', 'stgat_v2', 'checkpoints', 'stgat_resilient_best.pth'))

if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints'), exist_ok=True)
    train_resilient_stgat()
