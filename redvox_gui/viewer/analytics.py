"""
Advanced analytics for RedVox sensor data.
Provides cross-sensor correlation analysis, baseline drift detection, and threshold alerts.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional
import io
import base64


class CrossSensorCorrelationAnalyzer:
    """Analyze correlations between different sensors."""
    
    def __init__(self):
        self.correlation_matrix = None
        self.correlation_type = 'pearson'  # pearson, spearman
    
    def calculate_correlation(self, sensor_data: Dict[str, np.ndarray], 
                             method: str = 'pearson') -> Dict[str, np.ndarray]:
        """
        Calculate correlation matrix between sensors.
        
        Args:
            sensor_data: Dictionary with sensor names and data arrays
            method: Correlation method ('pearson' or 'spearman')
        
        Returns:
            Dictionary with correlation matrix and p-values
        """
        self.correlation_type = method
        
        # Find minimum length for alignment
        min_length = min(len(data) for data in sensor_data.values() if len(data) > 0)
        
        # Align all sensor data to same length
        aligned_data = {}
        for name, data in sensor_data.items():
            if len(data) >= min_length:
                aligned_data[name] = data[:min_length]
        
        # Create data matrix
        sensor_names = list(aligned_data.keys())
        n_sensors = len(sensor_names)
        
        if n_sensors < 2:
            return {'error': 'Need at least 2 sensors for correlation analysis'}
        
        data_matrix = np.array([aligned_data[name] for name in sensor_names])
        
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = np.corrcoef(data_matrix)
        elif method == 'spearman':
            corr_matrix, _ = spearmanr(data_matrix.T)
        else:
            return {'error': f'Unknown correlation method: {method}'}
        self.correlation_matrix = corr_matrix
        
        return {
            'correlation_matrix': corr_matrix,
            'sensor_names': sensor_names,
            'method': method
        }
    
    def generate_heatmap(self, correlation_data: Dict, figsize=(10, 8)) -> str:
        """
        Generate correlation heatmap visualization.
        
        Args:
            correlation_data: Correlation data from calculate_correlation
            figsize: Figure size
        
        Returns:
            Base64 encoded PNG image
        """
        if 'error' in correlation_data:
            return None
        
        corr_matrix = correlation_data['correlation_matrix']
        sensor_names = correlation_data['sensor_names']
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='#1e1e2e')
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=sensor_names,
            yticklabels=sensor_names,
            ax=ax,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        ax.set_title(f'Sensor Correlation Matrix ({correlation_data["method"]})', 
                    color='white', fontsize=14, pad=20)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        # Colorbar text color
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('Correlation Coefficient', color='white')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#1e1e2e')
        buf.seek(0)
        b64_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return b64_str
    
    def get_strong_correlations(self, threshold: float = 0.7) -> List[Dict]:
        """
        Get pairs of sensors with strong correlations.
        
        Args:
            threshold: Correlation threshold (absolute value)
        
        Returns:
            List of correlation pairs
        """
        if self.correlation_matrix is None:
            return []
        
        strong_correlations = []
        n = self.correlation_matrix.shape[0]
        
        for i in range(n):
            for j in range(i+1, n):
                corr = self.correlation_matrix[i, j]
                if abs(corr) >= threshold:
                    strong_correlations.append({
                        'sensor_1': f'Sensor_{i}',
                        'sensor_2': f'Sensor_{j}',
                        'correlation': float(corr),
                        'strength': 'strong' if abs(corr) >= 0.8 else 'moderate'
                    })
        
        return strong_correlations


class BaselineDriftDetector:
    """Detect baseline drift in sensor data."""
    
    def __init__(self):
        self.drift_threshold = 0.1  # 10% change considered drift
        self.window_size = 100  # Window for drift detection
    
    def detect_drift(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> Dict:
        """
        Detect baseline drift in sensor data.
        
        Args:
            data: Sensor data array
            timestamps: Optional timestamp array
        
        Returns:
            Dictionary with drift information
        """
        if len(data) < self.window_size:
            return {'error': 'Data too short for drift detection'}
        
        # Calculate moving average
        window = np.ones(self.window_size) / self.window_size
        moving_avg = np.convolve(data, window, mode='valid')
        
        # Calculate trend using linear regression
        x = np.arange(len(moving_avg))
        slope, intercept = np.polyfit(x, moving_avg, 1)
        
        # Calculate total drift
        initial_value = moving_avg[0]
        final_value = moving_avg[-1]
        total_drift = final_value - initial_value
        relative_drift = total_drift / (abs(initial_value) + 1e-10)
        
        # Detect drift segments
        drift_segments = []
        in_drift = False
        drift_start = 0
        
        for i in range(1, len(moving_avg)):
            change = abs(moving_avg[i] - moving_avg[i-1])
            avg_value = abs(moving_avg[i-1]) + 1e-10
            relative_change = change / avg_value
            
            if relative_change > self.drift_threshold and not in_drift:
                in_drift = True
                drift_start = i
            elif relative_change <= self.drift_threshold and in_drift:
                in_drift = False
                if timestamps is not None:
                    drift_segments.append({
                        'start_time': timestamps[drift_start],
                        'end_time': timestamps[i],
                        'duration': timestamps[i] - timestamps[drift_start]
                    })
                else:
                    drift_segments.append({
                        'start_index': drift_start,
                        'end_index': i,
                        'duration': i - drift_start
                    })
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'total_drift': float(total_drift),
            'relative_drift': float(relative_drift),
            'drift_detected': abs(relative_drift) > self.drift_threshold,
            'drift_segments': drift_segments,
            'drift_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        }
    
    def correct_drift(self, data: np.ndarray, method: str = 'linear') -> np.ndarray:
        """
        Correct baseline drift from sensor data.
        
        Args:
            data: Sensor data array
            method: Correction method ('linear', 'polynomial')
        
        Returns:
            Drift-corrected data
        """
        x = np.arange(len(data))
        
        if method == 'linear':
            slope, intercept = np.polyfit(x, data, 1)
            trend = slope * x + intercept
            corrected = data - trend
        elif method == 'polynomial':
            coeffs = np.polyfit(x, data, 2)
            trend = np.polyval(coeffs, x)
            corrected = data - trend
        else:
            return data
        
        return corrected


class ThresholdAlertSystem:
    """Detect threshold exceedances in sensor data."""
    
    def __init__(self):
        self.thresholds = {}
        self.alerts = []
    
    def set_threshold(self, sensor_name: str, min_value: Optional[float] = None, 
                     max_value: Optional[float] = None):
        """
        Set threshold for a sensor.
        
        Args:
            sensor_name: Name of the sensor
            min_value: Minimum allowed value (None for no minimum)
            max_value: Maximum allowed value (None for no maximum)
        """
        self.thresholds[sensor_name] = {
            'min': min_value,
            'max': max_value
        }
    
    def check_thresholds(self, sensor_data: Dict[str, np.ndarray], 
                       timestamps: Optional[np.ndarray] = None) -> Dict[str, List]:
        """
        Check if sensor data exceeds thresholds.
        
        Args:
            sensor_data: Dictionary with sensor names and data arrays
            timestamps: Optional timestamp array
        
        Returns:
            Dictionary with alerts for each sensor
        """
        alerts = {}
        
        for sensor_name, data in sensor_data.items():
            if sensor_name not in self.thresholds:
                continue
            
            threshold = self.thresholds[sensor_name]
            sensor_alerts = []
            
            # Check minimum threshold
            if threshold['min'] is not None:
                min_violations = np.where(data < threshold['min'])[0]
                for idx in min_violations:
                    if timestamps is not None:
                        sensor_alerts.append({
                            'type': 'minimum_exceeded',
                            'timestamp': timestamps[idx],
                            'value': float(data[idx]),
                            'threshold': threshold['min']
                        })
                    else:
                        sensor_alerts.append({
                            'type': 'minimum_exceeded',
                            'index': int(idx),
                            'value': float(data[idx]),
                            'threshold': threshold['min']
                        })
            
            # Check maximum threshold
            if threshold['max'] is not None:
                max_violations = np.where(data > threshold['max'])[0]
                for idx in max_violations:
                    if timestamps is not None:
                        sensor_alerts.append({
                            'type': 'maximum_exceeded',
                            'timestamp': timestamps[idx],
                            'value': float(data[idx]),
                            'threshold': threshold['max']
                        })
                    else:
                        sensor_alerts.append({
                            'type': 'maximum_exceeded',
                            'index': int(idx),
                            'value': float(data[idx]),
                            'threshold': threshold['max']
                        })
            
            alerts[sensor_name] = sensor_alerts
        
        self.alerts = alerts
        return alerts
    
    def generate_alert_summary(self) -> Dict:
        """Generate summary of all alerts."""
        total_alerts = sum(len(alerts) for alerts in self.alerts.values())
        
        summary = {
            'total_alerts': total_alerts,
            'sensors_with_alerts': [sensor for sensor, alerts in self.alerts.items() if alerts],
            'alert_count_by_sensor': {
                sensor: len(alerts) for sensor, alerts in self.alerts.items()
            },
            'alert_types': {}
        }
        
        # Count alert types
        for sensor, alerts in self.alerts.items():
            for alert in alerts:
                alert_type = alert['type']
                if alert_type not in summary['alert_types']:
                    summary['alert_types'][alert_type] = 0
                summary['alert_types'][alert_type] += 1
        
        return summary


class AdvancedAnalytics:
    """Main class for advanced analytics operations."""
    
    def __init__(self):
        self.correlation_analyzer = CrossSensorCorrelationAnalyzer()
        self.drift_detector = BaselineDriftDetector()
        self.alert_system = ThresholdAlertSystem()
    
    def analyze_packet(self, packet) -> Dict:
        """
        Perform comprehensive analytics on RedVox packet.
        
        Args:
            packet: RedVox packet object
        
        Returns:
            Dictionary with all analytics results
        """
        from viewer.ml_integration import extract_sensor_data_for_ml
        
        # Extract sensor data
        sensor_data = extract_sensor_data_for_ml(packet)
        
        results = {
            'correlation_analysis': {},
            'drift_analysis': {},
            'threshold_alerts': {}
        }
        
        # Prepare correlation data
        correlation_data = {}
        for sensor_name, data in sensor_data.items():
            if isinstance(data, dict):
                # Handle XYZ sensors
                if 'x' in data:
                    correlation_data[f'{sensor_name}_x'] = data['x']
                if 'y' in data:
                    correlation_data[f'{sensor_name}_y'] = data['y']
                if 'z' in data:
                    correlation_data[f'{sensor_name}_z'] = data['z']
                if 'samples' in data:
                    correlation_data[sensor_name] = data['samples']
            else:
                correlation_data[sensor_name] = data
        
        # Correlation analysis
        if len(correlation_data) >= 2:
            corr_results = self.correlation_analyzer.calculate_correlation(correlation_data)
            if 'error' not in corr_results:
                results['correlation_analysis'] = {
                    'matrix': corr_results['correlation_matrix'].tolist(),
                    'sensor_names': corr_results['sensor_names'],
                    'method': corr_results['method'],
                    'heatmap': self.correlation_analyzer.generate_heatmap(corr_results),
                    'strong_correlations': self.correlation_analyzer.get_strong_correlations()
                }
        
        # Drift analysis for each sensor
        for sensor_name, data in sensor_data.items():
            if isinstance(data, dict):
                if 'samples' in data:
                    drift_results = self.drift_detector.detect_drift(data['samples'])
                    results['drift_analysis'][sensor_name] = drift_results
            else:
                drift_results = self.drift_detector.detect_drift(data)
                results['drift_analysis'][sensor_name] = drift_results
        
        return results
