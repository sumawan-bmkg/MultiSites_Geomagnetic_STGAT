"""
Signal Processing Module
Bandpass filter PC3 dan analisis rasio Z/H untuk data geomagnetik.
"""
import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GeomagneticSignalProcessor:
    """Process geomagnetic signals with PC3 bandpass filter."""
    
    def __init__(self, sampling_rate=1.0):
        """
        Initialize processor.
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 1.0 for 1-second data)
        """
        self.fs = sampling_rate
        
        # PC3 pulsation frequency range: 22-100 mHz (0.022-0.1 Hz)
        # Period: 10-45 seconds
        self.pc3_low = 0.022  # Hz
        self.pc3_high = 0.1   # Hz
    
    def handle_gaps(self, data):
        """
        New Hybrid Gap Strategy for Dissertation Audit:
        - Gap <= 5s: Cubic Spline Interpolation.
        - Gap > 5s: Zero-Padding + Binary Mask.
        """
        valid_mask = ~np.isnan(data)
        data_clean = np.array(data, dtype=float)
        binary_mask = np.ones_like(data)
        
        if not np.any(~valid_mask):
            return data_clean, binary_mask

        # Detect gaps
        from scipy.ndimage import label
        missing_indices = np.where(~valid_mask)[0]
        labeled_array, num_features = label(~valid_mask)
        
        for i in range(1, num_features + 1):
            gap_indices = np.where(labeled_array == i)[0]
            gap_len = len(gap_indices)
            
            if gap_len <= 5:
                # Short gap: Cubic Spline
                # Need some neighbors for spline context
                start, end = gap_indices[0], gap_indices[-1]
                ctx_start = max(0, start - 10)
                ctx_end = min(len(data), end + 11)
                
                # Get valid points in context
                x_known = np.where(valid_mask[ctx_start:ctx_end])[0] + ctx_start
                y_known = data[x_known]
                
                if len(x_known) > 2:
                    cs = interpolate.CubicSpline(x_known, y_known)
                    data_clean[gap_indices] = cs(gap_indices)
                else:
                    # Fallback to linear if not enough points
                    data_clean[gap_indices] = np.interp(gap_indices, x_known, y_known)
            else:
                # Long gap: Zero-Padding + Masking
                data_clean[gap_indices] = 0.0
                binary_mask[gap_indices] = 0.0
                
        return data_clean, binary_mask

    def bandpass_filter(self, data, low_freq=None, high_freq=None, order=4):
        """
        Modified bandpass filter to return (filtered_data, binary_mask).
        """
        if low_freq is None: low_freq = self.pc3_low
        if high_freq is None: high_freq = self.pc3_high
        
        # 1. Handle Gaps with Hybrid Strategy
        data_clean, binary_mask = self.handle_gaps(data)
        
        # 2. Design Butterworth bandpass filter
        nyquist = self.fs / 2.0
        low = max(0.001, min(low_freq / nyquist, 0.999))
        high = max(low + 0.001, min(high_freq / nyquist, 0.999))
        
        try:
            b, a = signal.butter(order, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data_clean)
            return filtered, binary_mask
        except Exception as e:
            logger.error(f"Filter error: {e}")
            return data_clean, binary_mask
    
    def calculate_zh_ratio(self, z_data, h_data):
        """
        Calculate Z/H ratio.
        
        Args:
            z_data: Z component array
            h_data: H component array
            
        Returns:
            Z/H ratio array
        """
        # Avoid division by zero
        h_safe = np.where(np.abs(h_data) < 1e-10, 1e-10, h_data)
        ratio = z_data / h_safe
        
        return ratio
    
    def process_components(self, h_data, d_data, z_data, apply_pc3=True):
        """
        Process components and return masks.
        """
        result = {
            'h_raw': h_data,
            'd_raw': d_data,
            'z_raw': z_data,
            'mask': np.ones_like(h_data) # Global mask
        }
        
        if apply_pc3:
            h_filt, h_mask = self.bandpass_filter(h_data)
            d_filt, d_mask = self.bandpass_filter(d_data)
            z_filt, z_mask = self.bandpass_filter(z_data)
            
            result['h_pc3'] = h_filt
            result['d_pc3'] = d_filt
            result['z_pc3'] = z_filt
            
            # Combine masks: if any channel is invalid, the sample at that time is invalid
            result['mask'] = h_mask * d_mask * z_mask
            
            result['zh_ratio_pc3'] = self.calculate_zh_ratio(result['z_pc3'], result['h_pc3'])
        
        result['zh_ratio_raw'] = self.calculate_zh_ratio(z_data, h_data)
        return result
    
    def plot_raw_components(self, raw_data, title=None, save_path=None):
        """
        Plot three components (H, D, Z) raw data for 24-hour period.
        Display raw data as-is without any Y-axis limits (unlimited auto-scale).
        
        Args:
            raw_data: Dict with 'h_raw', 'd_raw', 'z_raw' keys
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        try:
            fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
            
            # Create time axis for 24 hours (86400 seconds)
            num_samples = len(raw_data['h_raw'])
            time_hours = np.arange(num_samples) / 3600.0  # Convert seconds to hours
            
            components = [
                ('h_raw', 'H Component (Northward)', 'red'),
                ('d_raw', 'D Component (Eastward)', 'green'),
                ('z_raw', 'Z Component (Vertical)', 'blue')
            ]
            
            for idx, (key, label, color) in enumerate(components):
                ax = axes[idx]
                data = raw_data[key]
                
                # Filter out NaN values for statistics
                valid_data = data[~np.isnan(data)]
                
                # Plot raw data with thin line for 24-hour visibility
                ax.plot(time_hours, data, 
                       color=color, linewidth=0.8, label='Raw Data', alpha=0.9)
                
                # Add mean line
                if len(valid_data) > 0:
                    mean_val = np.nanmean(data)
                    ax.axhline(y=mean_val, color=color, linestyle='--', 
                              linewidth=1.5, alpha=0.6, label=f'Mean: {mean_val:.2f} nT')
                
                ax.set_ylabel(f'{label}\n(nT)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                ax.legend(loc='upper right', fontsize=10)
                
                # NO Y-axis limit - let matplotlib auto-scale
                # This allows raw data to be displayed naturally
                
                # Add statistics box
                if len(valid_data) > 0:
                    raw_mean = np.nanmean(data)
                    raw_std = np.nanstd(data)
                    raw_min = np.nanmin(data)
                    raw_max = np.nanmax(data)
                    stats_text = f'Mean: {raw_mean:.3f} nT\nStd: {raw_std:.3f} nT\nMin: {raw_min:.3f} nT\nMax: {raw_max:.3f} nT'
                else:
                    stats_text = 'No valid data'
                
                ax.text(0.02, 0.98, stats_text, 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Set x-axis to show 24-hour period
            axes[-1].set_xlabel('Time (hours from 00:00)', fontsize=12, fontweight='bold')
            axes[-1].set_xlim(0, 24)
            axes[-1].set_xticks(np.arange(0, 25, 2))
            
            if title:
                fig.suptitle(title, fontsize=15, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Raw components plot saved to {save_path}")
            
            plt.close(fig)  # Close figure to free memory
            return fig
        except Exception as e:
            logger.error(f"Error plotting raw components: {e}")
            plt.close('all')
            raise
    
    def plot_raw_xy_components(self, raw_data, title=None, save_path=None):
        """
        Plot X and Y components (Cartesian coordinates) raw data for 24-hour period.
        Display raw data as-is without any Y-axis limits (unlimited auto-scale).
        
        X = H * cos(D_rad) - Northward component
        Y = H * sin(D_rad) - Eastward component
        
        Args:
            raw_data: Dict with 'h_raw', 'd_raw', 'z_raw' keys (X/Y computed from H/D)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        try:
            fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
            
            # Create time axis for 24 hours (86400 seconds)
            num_samples = len(raw_data['h_raw'])
            time_hours = np.arange(num_samples) / 3600.0  # Convert seconds to hours
            
            # Compute X and Y components from H and D
            h_data = raw_data['h_raw']
            d_data = raw_data['d_raw']
            z_data = raw_data['z_raw']
            
            # X = H * cos(D_rad), Y = H * sin(D_rad)
            d_rad = np.deg2rad(d_data)
            x_data = np.full_like(h_data, np.nan, dtype=np.float64)
            y_data = np.full_like(h_data, np.nan, dtype=np.float64)
            
            valid_idx = np.isfinite(h_data) & np.isfinite(d_rad)
            if np.any(valid_idx):
                x_data[valid_idx] = h_data[valid_idx] * np.cos(d_rad[valid_idx])
                y_data[valid_idx] = h_data[valid_idx] * np.sin(d_rad[valid_idx])
            
            components = [
                (x_data, 'X Component (Northward - Cartesian)', 'red'),
                (y_data, 'Y Component (Eastward - Cartesian)', 'green'),
                (z_data, 'Z Component (Vertical)', 'blue')
            ]
            
            for idx, (data, label, color) in enumerate(components):
                ax = axes[idx]
                
                # Filter out NaN values for statistics
                valid_data = data[~np.isnan(data)]
                
                # Plot raw data with thin line for 24-hour visibility
                ax.plot(time_hours, data, 
                       color=color, linewidth=0.8, label='Raw Data', alpha=0.9)
                
                # Add mean line
                if len(valid_data) > 0:
                    mean_val = np.nanmean(data)
                    ax.axhline(y=mean_val, color=color, linestyle='--', 
                              linewidth=1.5, alpha=0.6, label=f'Mean: {mean_val:.2f} nT')
                
                ax.set_ylabel(f'{label}\n(nT)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                ax.legend(loc='upper right', fontsize=10)
                
                # NO Y-axis limit - let matplotlib auto-scale
                # This allows raw data to be displayed naturally
                
                # Add statistics box
                if len(valid_data) > 0:
                    raw_mean = np.nanmean(data)
                    raw_std = np.nanstd(data)
                    raw_min = np.nanmin(data)
                    raw_max = np.nanmax(data)
                    stats_text = f'Mean: {raw_mean:.3f} nT\nStd: {raw_std:.3f} nT\nMin: {raw_min:.3f} nT\nMax: {raw_max:.3f} nT'
                else:
                    stats_text = 'No valid data'
                
                ax.text(0.02, 0.98, stats_text, 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Set x-axis to show 24-hour period
            axes[-1].set_xlabel('Time (hours from 00:00)', fontsize=12, fontweight='bold')
            axes[-1].set_xlim(0, 24)
            axes[-1].set_xticks(np.arange(0, 25, 2))
            
            if title:
                fig.suptitle(title, fontsize=15, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"X/Y components plot saved to {save_path}")
            
            plt.close(fig)  # Close figure to free memory
            return fig
        except Exception as e:
            logger.error(f"Error plotting X/Y components: {e}")
            plt.close('all')
            raise
    
    def plot_raw_zh_ratio(self, raw_data, title=None, save_path=None):
        """
        Plot Z/H ratio for raw data for 24-hour period.
        
        Args:
            raw_data: Dict with 'zh_ratio_raw' key
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
            
            # Create time axis for 24 hours
            num_samples = len(raw_data['zh_ratio_raw'])
            time_hours = np.arange(num_samples) / 3600.0  # Convert seconds to hours
            
            # Plot raw Z/H ratio
            ax.plot(time_hours, raw_data['zh_ratio_raw'], 
                    color='purple', linewidth=0.5, label='Raw Z/H Ratio', alpha=0.8)
            
            # Add mean line
            valid_ratio = raw_data['zh_ratio_raw'][~np.isnan(raw_data['zh_ratio_raw'])]
            if len(valid_ratio) > 0:
                mean_ratio = np.nanmean(raw_data['zh_ratio_raw'])
                ax.axhline(y=mean_ratio, color='purple', linestyle='--', 
                          linewidth=1, alpha=0.5, label=f'Mean: {mean_ratio:.3f}')
            
            ax.set_ylabel('Z/H Ratio', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (hours from 00:00)', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
            
            # NO Y-axis limit - let matplotlib auto-scale
            # This allows raw data to be displayed naturally like intermagnet
            
            # Set x-axis to show 24-hour period
            ax.set_xlim(0, 24)
            ax.set_xticks(np.arange(0, 25, 2))
            
            # Add statistics
            if len(valid_ratio) > 0:
                raw_mean = np.nanmean(raw_data['zh_ratio_raw'])
                raw_std = np.nanstd(raw_data['zh_ratio_raw'])
                raw_min = np.nanmin(raw_data['zh_ratio_raw'])
                raw_max = np.nanmax(raw_data['zh_ratio_raw'])
                stats_text = f'Mean: {raw_mean:.4f}\nStd: {raw_std:.4f}\nMin: {raw_min:.4f}\nMax: {raw_max:.4f}'
            else:
                stats_text = 'No valid data'
            
            ax.text(0.02, 0.98, stats_text, 
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            if title:
                fig.suptitle(title, fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Raw Z/H ratio plot saved to {save_path}")
            
            plt.close(fig)  # Close figure to free memory
            return fig
        except Exception as e:
            logger.error(f"Error plotting raw Z/H ratio: {e}")
            plt.close('all')
            raise

    def plot_components(self, processed_data, title=None, save_path=None):
        """
        Plot three components (H, D, Z) with PC3 filter.
        
        Args:
            processed_data: Dict from process_components()
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        time_axis = np.arange(len(processed_data['h_raw'])) / self.fs / 3600  # Hours
        
        components = [
            ('h_raw', 'h_pc3', 'H Component', 'red'),
            ('d_raw', 'd_pc3', 'D Component', 'green'),
            ('z_raw', 'z_pc3', 'Z Component', 'blue')
        ]
        
        for idx, (raw_key, pc3_key, label, color) in enumerate(components):
            ax = axes[idx]
            
            # Plot raw data (lighter)
            ax.plot(time_axis, processed_data[raw_key], 
                   color=color, alpha=0.3, linewidth=0.5, label='Raw')
            
            # Plot PC3 filtered (darker)
            if pc3_key in processed_data:
                ax.plot(time_axis, processed_data[pc3_key], 
                       color=color, linewidth=1.0, label='PC3 Filtered')
            
            ax.set_ylabel(f'{label}\n(nT)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            
            # Add statistics
            if pc3_key in processed_data:
                pc3_std = np.std(processed_data[pc3_key])
                ax.text(0.02, 0.95, f'PC3 σ: {pc3_std:.2f} nT', 
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[-1].set_xlabel('Time (hours)', fontsize=10)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_zh_ratio(self, processed_data, title=None, save_path=None):
        """
        Plot Z/H ratio comparison.
        
        Args:
            processed_data: Dict from process_components()
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        time_axis = np.arange(len(processed_data['zh_ratio_raw'])) / self.fs / 3600
        
        # Raw Z/H ratio
        axes[0].plot(time_axis, processed_data['zh_ratio_raw'], 
                    color='purple', linewidth=0.5, alpha=0.6)
        axes[0].set_ylabel('Z/H Ratio\n(Raw)', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        # Add statistics
        raw_mean = np.mean(processed_data['zh_ratio_raw'])
        raw_std = np.std(processed_data['zh_ratio_raw'])
        axes[0].text(0.02, 0.95, f'Mean: {raw_mean:.3f}\nσ: {raw_std:.3f}', 
                    transform=axes[0].transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # PC3 filtered Z/H ratio
        if 'zh_ratio_pc3' in processed_data:
            axes[1].plot(time_axis, processed_data['zh_ratio_pc3'], 
                        color='darkviolet', linewidth=1.0)
            axes[1].set_ylabel('Z/H Ratio\n(PC3 Filtered)', fontsize=10)
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            
            # Add statistics
            pc3_mean = np.mean(processed_data['zh_ratio_pc3'])
            pc3_std = np.std(processed_data['zh_ratio_pc3'])
            axes[1].text(0.02, 0.95, f'Mean: {pc3_mean:.3f}\nσ: {pc3_std:.3f}', 
                        transform=axes[1].transAxes, fontsize=8,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[-1].set_xlabel('Time (hours)', fontsize=10)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Z/H ratio plot saved to {save_path}")
        
        return fig
    
    def calculate_power_spectrum(self, data, nperseg=1024):
        """
        Calculate power spectral density.
        
        Args:
            data: Input signal
            nperseg: Length of each segment
            
        Returns:
            frequencies, power spectral density
        """
        # Remove NaN
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return np.array([]), np.array([])
        
        data_clean = np.array(data, dtype=float)
        if np.any(~valid_mask):
            x = np.arange(len(data))
            data_clean[~valid_mask] = np.interp(x[~valid_mask], x[valid_mask], data[valid_mask])
        
        # Calculate PSD
        freqs, psd = signal.welch(data_clean, fs=self.fs, nperseg=nperseg)
        
        return freqs, psd
    
    def plot_power_spectrum(self, processed_data, title=None, save_path=None):
        """
        Plot power spectrum for H, D, Z components.
        
        Args:
            processed_data: Dict from process_components()
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        components = [
            ('h_raw', 'H Component', 'red'),
            ('d_raw', 'D Component', 'green'),
            ('z_raw', 'Z Component', 'blue')
        ]
        
        for idx, (key, label, color) in enumerate(components):
            freqs, psd = self.calculate_power_spectrum(processed_data[key])
            
            if len(freqs) > 0:
                axes[idx].loglog(freqs, psd, color=color, linewidth=1.0)
                axes[idx].set_ylabel(f'{label}\nPSD', fontsize=10)
                axes[idx].grid(True, alpha=0.3, which='both')
                
                # Mark PC3 range
                axes[idx].axvspan(self.pc3_low, self.pc3_high, 
                                 alpha=0.2, color='yellow', 
                                 label='PC3 Range')
                axes[idx].legend(loc='upper right', fontsize=8)
        
        axes[-1].set_xlabel('Frequency (Hz)', fontsize=10)
        axes[-1].set_xlim(0.001, 0.5)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Power spectrum saved to {save_path}")
        
        return fig


if __name__ == '__main__':
    # Test signal processor
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate test signal
    t = np.linspace(0, 86400, 86400)  # 1 day, 1 Hz
    
    # Simulate geomagnetic data with PC3 pulsations
    h_test = 40000 + 100 * np.sin(2 * np.pi * 0.05 * t) + 20 * np.random.randn(len(t))
    d_test = 50 * np.cos(2 * np.pi * 0.03 * t) + 10 * np.random.randn(len(t))
    z_test = 30000 + 80 * np.sin(2 * np.pi * 0.04 * t) + 15 * np.random.randn(len(t))
    
    processor = GeomagneticSignalProcessor()
    
    # Process
    result = processor.process_components(h_test, d_test, z_test)
    
    # Plot
    processor.plot_components(result, title='Test Signal - PC3 Filter', 
                             save_path='test_pc3_components.png')
    processor.plot_zh_ratio(result, title='Test Signal - Z/H Ratio', 
                           save_path='test_zh_ratio.png')
    processor.plot_power_spectrum(result, title='Test Signal - Power Spectrum',
                                 save_path='test_power_spectrum.png')
    
    print("Test plots generated!")
