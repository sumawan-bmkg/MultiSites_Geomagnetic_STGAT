import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pywt

# Add project root to path
sys.path.append(os.getcwd())
from intial.geomagnetic_fetcher import GeomagneticDataFetcher
from intial.signal_processing import GeomagneticSignalProcessor
from src.preprocessing.cwt_processor import CWTProcessor

def visualize_one_day(date_str='2026-01-01', station='ALR'):
    fetcher = GeomagneticDataFetcher()
    processor = GeomagneticSignalProcessor(sampling_rate=1.0)
    cwt_proc = CWTProcessor()
    
    if not fetcher.connect():
        print("Failed to connect")
        return
        
    fetch_date = date_str.replace('-', '')
    stations = ['ALR', 'AMB', 'CLP', 'GSI', 'GTO', 'JYP', 'KPY', 'LPS', 'LUT', 
                'LWA', 'LWK', 'MLB', 'PLU', 'SBG', 'SCN', 'SKB', 'SMI', 'SRG', 
                'SRO', 'TND', 'TNT', 'TRD', 'TRT', 'YOG']
    
    print(f"Fetching data for {date_str}...")
    day_data = {}
    for s in stations:
        data = fetcher.fetch_data(date_str, s)
        if data and 'Hcomp' in data:
            h_filt, _ = processor.bandpass_filter(data['Hcomp'])
            day_data[s] = h_filt
    
    fetcher.disconnect()
    
    if station not in day_data:
        print(f"Target station {station} not found in fetched data.")
        return
        
    active_stations = sorted(day_data.keys())
    data_matrix = np.array([day_data[s] for s in active_stations])
    
    # Run PCA
    pca = PCA(n_components=1)
    X = data_matrix.T
    transformed = pca.fit_transform(X)
    global_noise = pca.inverse_transform(transformed).T
    
    target_idx = active_stations.index(station)
    raw_signal = data_matrix[target_idx]
    noise_signal = global_noise[target_idx]
    clean_signal = raw_signal - noise_signal
    
    # Generate Scalograms
    raw_scal = cwt_proc.generate_scalogram(raw_signal)
    clean_scal = cwt_proc.generate_scalogram(clean_signal)
    
    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=False)
    
    time_1d = np.arange(86400) / 3600.0 # Hours
    time_2d = np.arange(1440) / 60.0 # Hours
    
    # 1. Raw Signal
    axes[0].plot(time_1d, raw_signal, label='Raw Bandpassed (Pc3)', color='blue', alpha=0.7)
    axes[0].plot(time_1d, noise_signal, label='Global Noise (PC1)', color='red', alpha=0.8, linestyle='--')
    axes[0].set_title(f"Raw vs Global Noise - {station} - {date_str}")
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Cleaned Signal
    axes[1].plot(time_1d, clean_signal, label='Cleaned (Raw - PC1)', color='green')
    axes[1].set_title(f"Cleaned Signal (Local Component)")
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. Raw Scalogram
    im2 = axes[2].imshow(raw_scal, aspect='auto', extent=[0, 24, 0.022, 0.1], origin='lower', cmap='jet')
    axes[2].set_title("Raw Scalogram (Power)")
    fig.colorbar(im2, ax=axes[2], label='Power')
    
    # 4. Cleaned Scalogram
    im3 = axes[3].imshow(clean_scal, aspect='auto', extent=[0, 24, 0.022, 0.1], origin='lower', cmap='jet')
    axes[3].set_title("Cleaned Scalogram (Power)")
    fig.colorbar(im3, ax=axes[3], label='Power')
    
    plt.tight_layout()
    plot_path = f"denoising_check_{station}_{date_str}.png"
    plt.savefig(plot_path)
    print(f"Saved visualization to {plot_path}")

if __name__ == "__main__":
    visualize_one_day()
