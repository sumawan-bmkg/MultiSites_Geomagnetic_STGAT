import os
import sys
import h5py
import numpy as np
import pandas as pd
import pywt
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

# Add intial folder to path for imports
sys.path.append(os.path.join(os.getcwd(), 'intial'))
from geomagnetic_fetcher import GeomagneticDataFetcher
from signal_processing import GeomagneticSignalProcessor

# Configuration
HDF5_FILE = 'dataset_v13_train_val_M5_patched.h5'
STATIONS = ['ALR', 'AMB', 'CLP', 'GSI', 'GTO', 'JYP', 'KPY', 'LPS', 'LUT', 
            'LWA', 'LWK', 'MLB', 'PLU', 'SBG', 'SCN', 'SKB', 'SMI', 'SRG', 
            'SRO', 'TND', 'TNT', 'TRD', 'TRT', 'YOG']
NUM_SCALES = 128
FS = 1.0  # 1Hz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RawPipelineExecutor:
    def __init__(self):
        self.fetcher = GeomagneticDataFetcher()
        self.processor = GeomagneticSignalProcessor(sampling_rate=FS)
        
    def generate_scalogram(self, signal_data):
        """Generate CWT scalogram for a single component."""
        # Pc3 range: 0.022 - 0.1 Hz
        f_min, f_max = 0.022, 0.1
        
        try:
            f0 = pywt.central_frequency('morl')
        except AttributeError:
            # Fallback for older/newer pywt versions
            f0 = pywt.CentralFreq('morl')
        
        # Log-spaced frequencies to capture wide range of pulsations
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), NUM_SCALES)
        scales = f0 / (freqs * (1.0/FS))
        
        # Compute CWT
        coef, _ = pywt.cwt(signal_data, scales, 'morl', sampling_period=1.0/FS)
        
        # Magnitude and downsample to 1440 minutes
        mag = np.abs(coef)
        
        # Reshape to (128, 1440, 60) and mean over the last dimension
        # Shape of coef is (128, 86400)
        n_scales = mag.shape[0]
        n_samples = mag.shape[1]
        
        if n_samples == 86400:
            mag = mag.reshape(n_scales, 1440, 60).mean(axis=2)
        else:
            logger.warning(f"Unexpected signal length: {n_samples}. Resizing to 1440.")
            # Simple resizing/padding
            new_mag = np.zeros((n_scales, 1440))
            bins = np.linspace(0, n_samples, 1441).astype(int)
            for j in range(1440):
                if bins[j] < bins[j+1]:
                    new_mag[:, j] = mag[:, bins[j]:bins[j+1]].mean(axis=1)
            mag = new_mag
            
        return mag

    def process_day_station(self, date_str, station):
        """Fetch, process and generate tensors for a day-station pair."""
        try:
            # 1. Fetch data
            if self.fetcher.sftp_client is None:
                if not self.fetcher.connect():
                    return None
            
            # Use fetcher.fetch_data which returns components
            data = self.fetcher.fetch_data(date_str, station)
            if data is None:
                return None
            
            # Map components
            h_raw = data.get('Hcomp')
            d_raw = data.get('Dcomp')
            z_raw = data.get('Zcomp')
            
            if h_raw is None or d_raw is None or z_raw is None:
                return None
                
            # 2. Bandpass Filter (Pc3)
            # handle_gaps is called inside bandpass_filter
            h_filt, h_mask = self.processor.bandpass_filter(h_raw)
            d_filt, d_mask = self.processor.bandpass_filter(d_raw)
            z_filt, z_mask = self.processor.bandpass_filter(z_raw)
            
            # 3. Generate CWT Tensors
            h_scal = self.generate_scalogram(h_filt)
            d_scal = self.generate_scalogram(d_filt)
            z_scal = self.generate_scalogram(z_filt)
            
            # Stack into (3, 128, 1440)
            tensor = np.stack([h_scal, d_scal, z_scal], axis=0)
            return tensor.astype(np.float16)
            
        except Exception as e:
            logger.error(f"Error processing {date_str} {station}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def execute_pipeline(self, max_days=None):
        """Run the full replacement pipeline."""
        with h5py.File(HDF5_FILE, 'a') as f:
            for group_name in ['train', 'val']:
                if group_name not in f: continue
                group = f[group_name]
                dates = [d.decode() if isinstance(d, bytes) else d for d in group['dates'][:]]
                tensors = group['tensors'] # Shape: (days, 24, 3, 128, 1440)
                
                logger.info(f"Processing {group_name} set ({len(dates)} days)...")
                
                days_processed = 0
                for i, d_str in enumerate(tqdm(dates, desc=f"Updating {group_name}")):
                    if max_days and days_processed >= max_days:
                        break
                        
                    # Target 2018-2023 for replacement
                    year = int(d_str.split('-')[0])
                    if not (2018 <= year <= 2023):
                        continue
                        
                    logger.info(f"Day {i}: {d_str}")
                    success_stations = 0
                    for s_idx, station in enumerate(STATIONS):
                        tensor = self.process_day_station(d_str, station)
                        if tensor is not None:
                            tensors[i, s_idx] = tensor
                            success_stations += 1
                    
                    if success_stations > 0:
                        days_processed += 1
                
                # Label Sanitization
                self.sanitize_labels(group)
            
            # Dst Patching
            self.patch_dst(f['val'])

    def sanitize_labels(self, group):
        """Set y_event=0 if label_dist/azm are missing."""
        logger.info(f"Sanitizing labels for {group.name}...")
        y_event = group['label_event']
        dist = group['label_dist']
        azm = group['label_azm']
        
        modified_count = 0
        for i in range(len(y_event)):
            if y_event[i] == 1:
                # Check if all dists and azms for all stations are zero or NaN
                if np.all(dist[i] == 0) or np.all(azm[i] == 0):
                    y_event[i] = 0
                    modified_count += 1
        
        logger.info(f"Sanitized {modified_count} samples.")

    def patch_dst(self, val_group):
        """Patch missing Dst values in Val set using V9 dataset as source."""
        logger.info("Patching Dst values in Val set from V9 dataset...")
        cosmic = val_group['cosmic_features']
        dates = [d.decode() if isinstance(d, bytes) else d for d in val_group['dates'][:]]
        
        V9_FILE = 'scalogram_v9_multistation_graph.h5'
        if not os.path.exists(V9_FILE):
            logger.warning(f"{V9_FILE} not found. Skipping Dst patch.")
            return
            
        patched_count = 0
        with h5py.File(V9_FILE, 'r') as f_v9:
            v9_dates = [d.decode().split(' ')[0] if isinstance(d, bytes) else d.split(' ')[0] for d in f_v9['val']['dates'][:]]
            v9_cosmic = f_v9['val']['cosmic_features'][:]
            
            # Build lookup
            dst_lookup = {d: c[1] for d, c in zip(v9_dates, v9_cosmic)}
            
            for i, d_str in enumerate(dates):
                if cosmic[i, 1] == 0: # Dst is 0
                    if d_str in dst_lookup:
                        cosmic[i, 1] = dst_lookup[d_str]
                        patched_count += 1
        
        logger.info(f"Patched {patched_count} Dst values using V9 lookup.")

if __name__ == '__main__':
    executor = RawPipelineExecutor()
    # Run the full pipeline
    executor.execute_pipeline()
