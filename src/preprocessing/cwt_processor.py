import numpy as np
from sklearn.decomposition import PCA, FastICA
import pywt
import logging

logger = logging.getLogger(__name__)

class SpatialDenoiseProcessor:
    """
    Implements Spatio-Temporal De-noising using PCA or ICA across multiple stations.
    Designed to remove global common-mode noise (e.g., Solar Cycle / Storms) 
    while preserving local co-seismic signals.
    """
    
    def __init__(self, method='pca', n_components=None):
        self.method = method.lower()
        self.n_components = n_components
        
    def denoise_stations(self, data_matrix):
        """
        Denoise multiple stations using PCA/ICA.
        
        Args:
            data_matrix: np.ndarray of shape (N_stations, N_samples)
            
        Returns:
            denoised_matrix: np.ndarray of shape (N_stations, N_samples)
            noise_component: np.ndarray of shape (N_samples,) - The global noise removed
        """
        n_stations, n_samples = data_matrix.shape
        if n_stations < 2:
            return data_matrix, np.zeros(n_samples)
            
        # Transpose to (N_samples, N_stations) for sklearn
        X = data_matrix.T
        
        # Handle NaNs by simple interpolation or zeroing
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0.0)
            
        if self.method == 'pca':
            # PC1 is typically the global common-mode noise in geomagnetic networks
            pca = PCA(n_components=1)
            transformed = pca.fit_transform(X)
            global_noise = pca.inverse_transform(transformed)
            
            # Denoised = Original - Global Noise
            denoised = X - global_noise
            return denoised.T, global_noise.T[0]
            
        elif self.method == 'ica':
            # FastICA to separate independent components
            # We assume the most "global" component is the noise
            ica = FastICA(n_components=min(n_stations, 3), random_state=42)
            transformed = ica.fit_transform(X)
            
            # Find the component with the highest correlation across stations
            correlations = []
            for i in range(transformed.shape[1]):
                comp = transformed[:, i]
                mean_corr = np.mean([np.abs(np.corrcoef(comp, X[:, j])[0, 1]) for j in range(n_stations)])
                correlations.append(mean_corr)
            
            global_idx = np.argmax(correlations)
            
            # Reconstruct without the global component
            transformed_clean = transformed.copy()
            noise_comp = transformed_clean[:, global_idx].copy()
            transformed_clean[:, global_idx] = 0
            denoised = ica.inverse_transform(transformed_clean)
            
            return denoised.T, noise_comp
        
        else:
            raise ValueError(f"Unknown method: {self.method}")

class CWTProcessor:
    """
    Handles CWT generation with integrated spatial denoising.
    """
    
    def __init__(self, num_scales=128, fs=1.0):
        self.num_scales = num_scales
        self.fs = fs
        self.denoiser = SpatialDenoiseProcessor(method='pca')
        
    def generate_scalogram(self, signal_data):
        """Standard CWT scalogram generation logic from execute_raw_pipeline.py."""
        f_min, f_max = 0.022, 0.1
        try:
            f0 = pywt.central_frequency('morl')
        except AttributeError:
            f0 = pywt.CentralFreq('morl')
            
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), self.num_scales)
        scales = f0 / (freqs * (1.0/self.fs))
        
        coef, _ = pywt.cwt(signal_data, scales, 'morl', sampling_period=1.0/self.fs)
        mag = np.abs(coef).astype(np.float32)
        del coef
        
        # Resample to 1440 minutes
        n_scales, n_samples = mag.shape
        if n_samples == 86400:
            mag = mag.reshape(n_scales, 1440, 60).mean(axis=2)
        else:
            new_mag = np.zeros((n_scales, 1440))
            bins = np.linspace(0, n_samples, 1441).astype(int)
            for j in range(1440):
                if bins[j] < bins[j+1]:
                    new_mag[:, j] = mag[:, bins[j]:bins[j+1]].mean(axis=1)
            mag = new_mag
            
        return mag

    def process_day_full_network(self, stations_raw_dict, all_station_codes):
        """
        Process all stations for a single day with spatial de-noising.
        
        Args:
            stations_raw_dict: Dict mapping station_name -> { 'H': h_array, 'D': d_array, 'Z': z_array }
            all_station_codes: List of all 24 station codes to ensure consistent matrix shape
            
        Returns:
            scalograms_dict: Dict mapping station_name -> (3, 128, 1440) tensor
        """
        if not stations_raw_dict:
            return {}
            
        # 1. Find sample length from first available station
        first_station = next(iter(stations_raw_dict.values()))
        sample_len = len(first_station['H'])
        
        # 2. Prepare matrices for each component (including missing stations as zeros)
        n_sta = len(all_station_codes)
        h_matrix = np.zeros((n_sta, sample_len))
        d_matrix = np.zeros((n_sta, sample_len))
        z_matrix = np.zeros((n_sta, sample_len))
        
        active_mask = np.zeros(n_sta, dtype=bool)
        
        for i, s in enumerate(all_station_codes):
            if s in stations_raw_dict:
                h_matrix[i] = stations_raw_dict[s]['H']
                d_matrix[i] = stations_raw_dict[s]['D']
                z_matrix[i] = stations_raw_dict[s]['Z']
                active_mask[i] = True
            
        # 3. Apply Spatial De-noising
        # Note: We apply PCA on all stations. Missing stations (zeros) won't contribute
        # much to the common-mode noise calculation but will be "denoised" by subtracting
        # the global component, which might lead to artifacts. 
        # Better: Fit PCA only on active stations, then subtract from all.
        
        active_indices = np.where(active_mask)[0]
        if len(active_indices) < 2:
            logger.warning("Fewer than 2 active stations. Skipping spatial denoising.")
            h_clean, d_clean, z_clean = h_matrix, d_matrix, z_matrix
        else:
            h_clean = self._denoise_component(h_matrix, active_indices)
            d_clean = self._denoise_component(d_matrix, active_indices)
            z_clean = self._denoise_component(z_matrix, active_indices)
        
        # 4. Generate CWT for each cleaned signal
        results = {}
        for i, s in enumerate(all_station_codes):
            if active_mask[i]:
                h_scal = self.generate_scalogram(h_clean[i])
                d_scal = self.generate_scalogram(d_clean[i])
                z_scal = self.generate_scalogram(z_clean[i])
                results[s] = np.stack([h_scal, d_scal, z_scal], axis=0).astype(np.float16)
            else:
                # Still return zero tensor for consistency if requested, or just skip
                results[s] = np.zeros((3, self.num_scales, 1440), dtype=np.float16)
                
        return results

    def _denoise_component(self, matrix, active_indices):
        """Helper to apply PCA denoising using only active stations for fitting."""
        active_data = matrix[active_indices]
        # Fit PCA on active stations
        pca = PCA(n_components=1)
        pca.fit(active_data.T)
        
        # The first PC is the global noise
        global_noise = pca.inverse_transform(pca.transform(active_data.T)).T
        
        # Subtract this global noise from the active stations
        cleaned_active = active_data - global_noise
        
        # Reconstruct full matrix
        full_clean = np.zeros_like(matrix)
        full_clean[active_indices] = cleaned_active
        return full_clean
