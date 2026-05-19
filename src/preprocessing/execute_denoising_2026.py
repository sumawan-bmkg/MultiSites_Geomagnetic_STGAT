import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import logging

# Add project root and intial to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'intial'))

from intial.geomagnetic_fetcher import GeomagneticDataFetcher
from intial.signal_processing import GeomagneticSignalProcessor
from src.preprocessing.cwt_processor import CWTProcessor

# Configuration
SOURCE_HDF5 = 'dataset_v13_blindtest_M5_patched.h5'
OUTPUT_HDF5 = 'dataset_v13_blindtest_M5_denoised.h5'
STATIONS = ['ALR', 'AMB', 'CLP', 'GSI', 'GTO', 'JYP', 'KPY', 'LPS', 'LUT', 
            'LWA', 'LWK', 'MLB', 'PLU', 'SBG', 'SCN', 'SKB', 'SMI', 'SRG', 
            'SRO', 'TND', 'TNT', 'TRD', 'TRT', 'YOG']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def execute_denoising_pipeline(limit=None):
    fetcher = GeomagneticDataFetcher()
    processor = GeomagneticSignalProcessor(sampling_rate=1.0)
    cwt_proc = CWTProcessor(num_scales=128, fs=1.0)
    
    if not fetcher.connect():
        logger.error("Could not connect to geomagnetic server.")
        return

    try:
        # Check if output exists for resume
        processed_count = 0
        if os.path.exists(OUTPUT_HDF5):
            with h5py.File(OUTPUT_HDF5, 'r') as f:
                if 'test/tensors' in f:
                    # Count non-zero days
                    tensors = f['test/tensors']
                    for i in range(tensors.shape[0]):
                        if np.any(tensors[i] != 0):
                            processed_count = i + 1
                        else:
                            break
            logger.info(f"Resuming from day {processed_count}")

        with h5py.File(SOURCE_HDF5, 'r') as src:
            test_grp = src['test']
            dates = test_grp['dates'][:]
            dates_str = [d.decode() if isinstance(d, bytes) else d for d in dates]
            
            if limit:
                dates_str = dates_str[:limit]
            
            n_days = len(dates_str)
            n_sta = len(STATIONS)
            
            # Create/Open output HDF5
            mode = 'a' if os.path.exists(OUTPUT_HDF5) else 'w'
            with h5py.File(OUTPUT_HDF5, mode) as dst:
                if mode == 'w':
                    out_grp = dst.create_group('test')
                    # Copy attributes
                    for attr_name, attr_val in test_grp.attrs.items():
                        out_grp.attrs[attr_name] = attr_val
                    # Prepare datasets
                    tensors_ds = out_grp.create_dataset(
                        'tensors', shape=(n_days, n_sta, 3, 128, 1440),
                        dtype=np.float16, compression='gzip', chunks=(1, n_sta, 3, 128, 1440)
                    )
                    # Copy other labels
                    for label in ['label_event', 'label_mag', 'label_azm', 'label_dist', 'cosmic_features', 'dates']:
                        if label in test_grp:
                            data = test_grp[label][:]
                            if limit:
                                data = data[:limit]
                            out_grp.create_dataset(label, data=data)
                else:
                    out_grp = dst['test']
                    tensors_ds = out_grp['tensors']

                logger.info(f"Starting denoising for {n_days} days (from index {processed_count})...")
                
                for i in range(processed_count, n_days):
                    d_str = dates_str[i]
                    fetch_date = d_str
                    
                    day_data = {}
                    for station in STATIONS:
                        try:
                            data = fetcher.fetch_data(fetch_date, station)
                            if data and 'Hcomp' in data:
                                h_filt, _ = processor.bandpass_filter(data['Hcomp'])
                                d_filt, _ = processor.bandpass_filter(data['Dcomp'])
                                z_filt, _ = processor.bandpass_filter(data['Zcomp'])
                                day_data[station] = {'H': h_filt, 'D': d_filt, 'Z': z_filt}
                        except Exception as e:
                            logger.warning(f"Failed to fetch {station} for {d_str}: {e}")
                    
                    if not day_data:
                        logger.warning(f"No data fetched for {d_str}. Saving zeros.")
                        tensors_ds[i] = np.zeros((n_sta, 3, 128, 1440), dtype=np.float16)
                    else:
                        logger.info(f"Processing day {d_str} ({i+1}/{n_days}) with {len(day_data)} stations...")
                        day_scalograms = cwt_proc.process_day_full_network(day_data, STATIONS)
                        day_tensor = np.zeros((n_sta, 3, 128, 1440), dtype=np.float16)
                        for s_idx, s_code in enumerate(STATIONS):
                            if s_code in day_scalograms:
                                day_tensor[s_idx] = day_scalograms[s_code]
                        tensors_ds[i] = day_tensor
                    
                    # Flush to disk every day
                    dst.flush()
                    
    finally:
        fetcher.disconnect()
        logger.info("Pipeline execution finished.")

if __name__ == "__main__":
    # Execute full range for 2026
    execute_denoising_pipeline()
