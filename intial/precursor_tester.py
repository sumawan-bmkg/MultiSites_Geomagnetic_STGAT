"""
Precursor Tester - Test geomagnetic data with model classification
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from torchvision import transforms
from PIL import Image

from src.geomagnetic_fetcher import GeomagneticDataFetcher
from src.signal_processing import GeomagneticSignalProcessor


class PrecursorTester:
    """Test geomagnetic data for precursor classification"""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize tester
        
        Args:
            model: PyTorch model for classification
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.processor = GeomagneticSignalProcessor(sampling_rate=1.0)
        
        # Image transform for model input
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def test_date(self, station, date_str, output_dir='output/precursor_test'):
        """
        Test geomagnetic data for a specific date
        
        Args:
            station: Station code (e.g., 'GTO')
            date_str: Date string in YYYY-MM-DD format
            output_dir: Directory to save output files
            
        Returns:
            dict: Test results with prediction and metadata
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Fetch geomagnetic data
        with GeomagneticDataFetcher() as fetcher:
            geo_data = fetcher.fetch_data(date_str, station)
        
        if geo_data is None:
            raise ValueError(f"Failed to fetch data for {station} on {date_str}")
        
        # Process with PC3 filter
        processed = self.processor.process_components(
            geo_data['Hcomp'],
            geo_data['Dcomp'],
            geo_data['Zcomp'],
            apply_pc3=True
        )
        
        # Generate 3-component image
        image_path = self._generate_3component_image(
            processed, station, date_str, output_dir
        )
        
        # Generate Z/H ratio PC3 comparison plot (NEW)
        zh_ratio_path = self._generate_zh_ratio_pc3_plot(
            geo_data, processed, station, date_str, output_dir
        )
        
        # Run inference
        prediction = self._classify_image(image_path)
        
        # Prepare results
        results = {
            'station': station,
            'date': date_str,
            'image_path': image_path,
            'image_url': f'/output/{os.path.basename(output_dir)}/{os.path.basename(image_path)}',
            'zh_ratio_path': zh_ratio_path,  # NEW
            'zh_ratio_url': f'/output/{os.path.basename(output_dir)}/{os.path.basename(zh_ratio_path)}',  # NEW
            'prediction': prediction,
            'coverage': float(geo_data['stats']['coverage']),
            'valid_samples': int(geo_data['stats']['valid_samples']),
            'h_mean': float(geo_data['stats']['h_mean']),
            'h_std': float(geo_data['stats']['h_std']),
            'z_mean': float(geo_data['stats']['z_mean']),
            'z_std': float(geo_data['stats']['z_std']),
        }
        
        return results
    
    def _generate_3component_image(self, processed, station, date_str, output_dir):
        """Generate 3-component image for model input"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        fig.patch.set_facecolor('black')
        
        time_hours = np.arange(len(processed['h_pc3'])) / 3600.0
        
        # H component
        axes[0].plot(time_hours, processed['h_pc3'], 'r-', linewidth=0.5)
        axes[0].set_ylabel('H (nT)', color='white')
        axes[0].set_title(f'{station} - {date_str} - PC3 Filtered', color='white')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_facecolor('black')
        axes[0].tick_params(colors='white')
        axes[0].spines['bottom'].set_color('white')
        axes[0].spines['top'].set_color('white')
        axes[0].spines['left'].set_color('white')
        axes[0].spines['right'].set_color('white')
        
        # D component
        axes[1].plot(time_hours, processed['d_pc3'], 'g-', linewidth=0.5)
        axes[1].set_ylabel('D (nT)', color='white')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_facecolor('black')
        axes[1].tick_params(colors='white')
        axes[1].spines['bottom'].set_color('white')
        axes[1].spines['top'].set_color('white')
        axes[1].spines['left'].set_color('white')
        axes[1].spines['right'].set_color('white')
        
        # Z component
        axes[2].plot(time_hours, processed['z_pc3'], 'b-', linewidth=0.5)
        axes[2].set_ylabel('Z (nT)', color='white')
        axes[2].set_xlabel('Time (hours)', color='white')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_facecolor('black')
        axes[2].tick_params(colors='white')
        axes[2].spines['bottom'].set_color('white')
        axes[2].spines['top'].set_color('white')
        axes[2].spines['left'].set_color('white')
        axes[2].spines['right'].set_color('white')
        
        plt.tight_layout()
        
        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f'{station}_{date_str.replace("-", "")}_{timestamp}.png'
        image_path = os.path.join(output_dir, image_filename)
        plt.savefig(image_path, dpi=100, facecolor='black')
        plt.close()
        
        return image_path
    
    def _generate_zh_ratio_pc3_plot(self, geo_data, processed, station, date_str, output_dir):
        """
        Generate Z/H ratio comparison plot: Raw vs PC3 Filtered (NEW)
        
        Args:
            geo_data: Raw geomagnetic data from fetcher
            processed: Processed data with PC3 filter applied
            station: Station code
            date_str: Date string
            output_dir: Output directory
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        time_hours = np.arange(len(geo_data['Zcomp'])) / 3600.0
        
        # Calculate Z/H ratios
        z_raw = geo_data['Zcomp']
        h_raw = geo_data['Hcomp']
        z_pc3 = processed['z_raw']
        h_pc3 = processed['h_raw']
        
        # Avoid division by zero
        h_raw_safe = np.where(np.abs(h_raw) < 1e-10, np.nan, h_raw)
        h_pc3_safe = np.where(np.abs(h_pc3) < 1e-10, np.nan, h_pc3)
        
        zh_raw = z_raw / h_raw_safe
        zh_pc3 = z_pc3 / h_pc3_safe
        
        # Plot 1: Raw Z/H Ratio
        axes[0].plot(time_hours, zh_raw, 'purple', linewidth=0.8, alpha=0.8, label='Raw Z/H')
        axes[0].axhline(y=np.nanmean(zh_raw), color='purple', linestyle='--', 
                       linewidth=1.5, alpha=0.6, label=f'Mean: {np.nanmean(zh_raw):.3f}')
        axes[0].set_ylabel('Z/H Ratio', fontsize=11, fontweight='bold')
        axes[0].set_title(f'{station} - {date_str} - Z/H Ratio Comparison', 
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].set_xlim(0, 24)
        
        # Add statistics box for raw
        raw_stats = f'Mean: {np.nanmean(zh_raw):.4f}\nStd: {np.nanstd(zh_raw):.4f}\nMin: {np.nanmin(zh_raw):.4f}\nMax: {np.nanmax(zh_raw):.4f}'
        axes[0].text(0.02, 0.98, raw_stats, transform=axes[0].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Plot 2: PC3 Filtered Z/H Ratio
        axes[1].plot(time_hours, zh_pc3, 'orange', linewidth=0.8, alpha=0.8, label='PC3 Filtered Z/H')
        axes[1].axhline(y=np.nanmean(zh_pc3), color='orange', linestyle='--',
                       linewidth=1.5, alpha=0.6, label=f'Mean: {np.nanmean(zh_pc3):.3f}')
        axes[1].set_ylabel('Z/H Ratio', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Time (hours from 00:00)', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].set_xlim(0, 24)
        axes[1].set_xticks(np.arange(0, 25, 2))
        
        # Add statistics box for PC3
        pc3_stats = f'Mean: {np.nanmean(zh_pc3):.4f}\nStd: {np.nanstd(zh_pc3):.4f}\nMin: {np.nanmin(zh_pc3):.4f}\nMax: {np.nanmax(zh_pc3):.4f}'
        axes[1].text(0.02, 0.98, pc3_stats, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'{station}_{date_str.replace("-", "")}_zh_ratio_pc3_{timestamp}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _classify_image(self, image_path):
        """Classify 3-component image with model"""
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
        
        # Map prediction to class name
        class_names = ['Normal', 'Precursor']
        pred_class = class_names[pred.item()]
        conf_value = float(confidence.item())
        
        return {
            'class': pred_class,
            'confidence': conf_value,
            'class_id': int(pred.item())
        }


if __name__ == '__main__':
    # Example usage
    from src.model import EarthquakeCNN
    
    # Load model
    model = EarthquakeCNN(num_classes=2)
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    
    # Test
    tester = PrecursorTester(model, device='cpu')
    results = tester.test_date('GTO', '2024-01-15')
    
    print(f"Station: {results['station']}")
    print(f"Date: {results['date']}")
    print(f"Prediction: {results['prediction']['class']}")
    print(f"Confidence: {results['prediction']['confidence']:.2%}")
    print(f"Coverage: {results['coverage']:.1f}%")
