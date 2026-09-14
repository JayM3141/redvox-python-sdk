"""
3D spatial visualization for RedVox sensor data.
Provides interactive 3D visualization for accelerometer, gyroscope, magnetometer, and location data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple
import io
import base64


class Visualizer3D:
    """Base class for 3D visualization."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.fig = None
        self.ax = None
    
    def setup_plot(self, title: str = "3D Visualization"):
        """Setup 3D plot with dark theme."""
        self.fig = plt.figure(figsize=self.figsize, facecolor='#1e1e2e')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1e1e2e')
        self.ax.set_title(title, color='white', fontsize=14)
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        
        # Set pane colors
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.grid(True, alpha=0.3)
    
    def to_base64(self) -> str:
        """Convert plot to base64 string."""
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#1e1e2e')
        buf.seek(0)
        b64_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(self.fig)
        return b64_str


class Accelerometer3DVisualizer(Visualizer3D):
    """3D visualization for accelerometer data."""
    
    def visualize(self, accel_data: Dict[str, np.ndarray], 
                  timestamps: Optional[np.ndarray] = None,
                  show_trajectory: bool = True,
                  show_vectors: bool = True) -> str:
        """
        Visualize accelerometer data in 3D.
        
        Args:
            accel_data: Dictionary with 'x', 'y', 'z' arrays
            timestamps: Optional timestamp array
            show_trajectory: Show 3D trajectory
            show_vectors: Show vector field
        
        Returns:
            Base64 encoded PNG image
        """
        x = accel_data['x']
        y = accel_data['y']
        z = accel_data['z']
        
        self.setup_plot("Accelerometer 3D Visualization")
        
        # Color by time if timestamps provided
        if timestamps is not None and len(timestamps) == len(x):
            colors = timestamps
            cmap = plt.cm.viridis
        else:
            colors = np.linspace(0, 1, len(x))
            cmap = plt.cm.plasma
        
        # Show trajectory
        if show_trajectory:
            self.ax.plot(x, y, z, color='#4ecdc4', linewidth=1, alpha=0.7, label='Trajectory')
            
            # Color-coded points
            scatter = self.ax.scatter(x, y, z, c=colors, cmap=cmap, s=10, alpha=0.6)
            if timestamps is not None:
                cbar = self.fig.colorbar(scatter, ax=self.ax, pad=0.1)
                cbar.set_label('Time', color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Show vector field (subsampled)
        if show_vectors and len(x) > 10:
            step = max(1, len(x) // 50)  # Show at most 50 vectors
            self.ax.quiver(
                np.zeros(len(x[::step])),
                np.zeros(len(y[::step])),
                np.zeros(len(z[::step])),
                x[::step], y[::step], z[::step],
                color='#ff6b6b', alpha=0.5, length=0.1, normalize=True,
                label='Vectors'
            )
        
        # Set labels
        self.ax.set_xlabel('X (m/s²)', color='white')
        self.ax.set_ylabel('Y (m/s²)', color='white')
        self.ax.set_zlabel('Z (m/s²)', color='white')
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.ax.legend()
        
        return self.to_base64()


class Gyroscope3DVisualizer(Visualizer3D):
    """3D visualization for gyroscope data."""
    
    def visualize(self, gyro_data: Dict[str, np.ndarray],
                  timestamps: Optional[np.ndarray] = None,
                  show_trajectory: bool = True,
                  show_rotation_path: bool = True) -> str:
        """
        Visualize gyroscope data in 3D.
        
        Args:
            gyro_data: Dictionary with 'x', 'y', 'z' arrays
            timestamps: Optional timestamp array
            show_trajectory: Show 3D trajectory
            show_rotation_path: Show rotation path
        
        Returns:
            Base64 encoded PNG image
        """
        x = gyro_data['x']
        y = gyro_data['y']
        z = gyro_data['z']
        
        self.setup_plot("Gyroscope 3D Visualization")
        
        # Color by time if timestamps provided
        if timestamps is not None and len(timestamps) == len(x):
            colors = timestamps
            cmap = plt.cm.viridis
        else:
            colors = np.linspace(0, 1, len(x))
            cmap = plt.cm.plasma
        
        # Show trajectory
        if show_trajectory:
            self.ax.plot(x, y, z, color='#45b7d1', linewidth=1, alpha=0.7, label='Angular Velocity')
            
            # Color-coded points
            scatter = self.ax.scatter(x, y, z, c=colors, cmap=cmap, s=10, alpha=0.6)
            if timestamps is not None:
                cbar = self.fig.colorbar(scatter, ax=self.ax, pad=0.1)
                cbar.set_label('Time', color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Show rotation path (integrated angles)
        if show_rotation_path and len(x) > 1:
            # Simple integration to show rotation
            dt = 1.0 if timestamps is None else np.mean(np.diff(timestamps))
            angle_x = np.cumsum(x) * dt
            angle_y = np.cumsum(y) * dt
            angle_z = np.cumsum(z) * dt
            
            self.ax.plot(angle_x, angle_y, angle_z, color='#f7b731', linewidth=2, 
                        alpha=0.8, linestyle='--', label='Integrated Rotation')
        
        # Set labels
        self.ax.set_xlabel('X (rad/s)', color='white')
        self.ax.set_ylabel('Y (rad/s)', color='white')
        self.ax.set_zlabel('Z (rad/s)', color='white')
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.ax.legend()
        
        return self.to_base64()


class Magnetometer3DVisualizer(Visualizer3D):
    """3D visualization for magnetometer data."""
    
    def visualize(self, mag_data: Dict[str, np.ndarray],
                  timestamps: Optional[np.ndarray] = None,
                  show_field_lines: bool = True,
                  show_compass: bool = True) -> str:
        """
        Visualize magnetometer data in 3D.
        
        Args:
            mag_data: Dictionary with 'x', 'y', 'z' arrays
            timestamps: Optional timestamp array
            show_field_lines: Show magnetic field lines
            show_compass: Show compass direction
        
        Returns:
            Base64 encoded PNG image
        """
        x = mag_data['x']
        y = mag_data['y']
        z = mag_data['z']
        
        self.setup_plot("Magnetometer 3D Visualization")
        
        # Color by time if timestamps provided
        if timestamps is not None and len(timestamps) == len(x):
            colors = timestamps
            cmap = plt.cm.viridis
        else:
            colors = np.linspace(0, 1, len(x))
            cmap = plt.cm.plasma
        
        # Show field trajectory
        if show_field_lines:
            self.ax.plot(x, y, z, color='#a55eea', linewidth=1, alpha=0.7, label='Magnetic Field')
            
            # Color-coded points
            scatter = self.ax.scatter(x, y, z, c=colors, cmap=cmap, s=10, alpha=0.6)
            if timestamps is not None:
                cbar = self.fig.colorbar(scatter, ax=self.ax, pad=0.1)
                cbar.set_label('Time', color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Show compass direction (average direction)
        if show_compass:
            avg_x = np.mean(x)
            avg_y = np.mean(y)
            avg_z = np.mean(z)
            
            # Draw compass needle
            self.ax.quiver(0, 0, 0, avg_x, avg_y, avg_z, 
                          color='#fd9644', linewidth=3, arrow_length_ratio=0.2,
                          label='Average Direction')
        
        # Set labels
        self.ax.set_xlabel('X (μT)', color='white')
        self.ax.set_ylabel('Y (μT)', color='white')
        self.ax.set_zlabel('Z (μT)', color='white')
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.ax.legend()
        
        return self.to_base64()


class Location3DVisualizer(Visualizer3D):
    """3D visualization for location data with altitude."""
    
    def visualize(self, location_data: Dict[str, np.ndarray],
                  timestamps: Optional[np.ndarray] = None,
                  show_trajectory: bool = True,
                  show_altitude_profile: bool = True) -> str:
        """
        Visualize location data in 3D with altitude.
        
        Args:
            location_data: Dictionary with 'latitude', 'longitude', 'altitude' arrays
            timestamps: Optional timestamp array
            show_trajectory: Show 3D trajectory
            show_altitude_profile: Show altitude profile
        
        Returns:
            Base64 encoded PNG image
        """
        lat = location_data['latitude']
        lon = location_data['longitude']
        alt = location_data.get('altitude', np.zeros_like(lat))
        
        self.setup_plot("Location 3D Visualization")
        
        # Color by time if timestamps provided
        if timestamps is not None and len(timestamps) == len(lat):
            colors = timestamps
            cmap = plt.cm.viridis
        else:
            colors = np.linspace(0, 1, len(lat))
            cmap = plt.cm.plasma
        
        # Show 3D trajectory
        if show_trajectory:
            self.ax.plot(lon, lat, alt, color='#26de81', linewidth=2, alpha=0.8, label='Trajectory')
            
            # Color-coded points
            scatter = self.ax.scatter(lon, lat, alt, c=colors, cmap=cmap, s=20, alpha=0.7)
            if timestamps is not None:
                cbar = self.fig.colorbar(scatter, ax=self.ax, pad=0.1)
                cbar.set_label('Time', color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Show start and end points
        if len(lat) > 0:
            self.ax.scatter(lon[0], lat[0], alt[0], color='#fc5c65', s=100, 
                          marker='o', label='Start', zorder=10)
            self.ax.scatter(lon[-1], lat[-1], alt[-1], color='#4b7bec', s=100, 
                          marker='s', label='End', zorder=10)
        
        # Set labels
        self.ax.set_xlabel('Longitude', color='white')
        self.ax.set_ylabel('Latitude', color='white')
        self.ax.set_zlabel('Altitude (m)', color='white')
        
        # Set view angle for better 3D perspective
        self.ax.view_init(elev=30, azim=45)
        
        self.ax.legend()
        
        return self.to_base64()


class MultiSensor3DVisualizer(Visualizer3D):
    """3D visualization combining multiple sensors."""
    
    def visualize_sensor_fusion(self, accel_data: Dict[str, np.ndarray],
                                gyro_data: Dict[str, np.ndarray],
                                mag_data: Dict[str, np.ndarray],
                                timestamps: Optional[np.ndarray] = None) -> str:
        """
        Visualize sensor fusion in 3D.
        
        Args:
            accel_data: Accelerometer data
            gyro_data: Gyroscope data
            mag_data: Magnetometer data
            timestamps: Optional timestamp array
        
        Returns:
            Base64 encoded PNG image
        """
        self.setup_plot("Multi-Sensor Fusion 3D Visualization")
        
        # Normalize data for comparison
        def normalize(data):
            x = data['x']
            y = data['y']
            z = data['z']
            mag = np.sqrt(x**2 + y**2 + z**2)
            mag[mag == 0] = 1  # Avoid division by zero
            return {
                'x': x / mag,
                'y': y / mag,
                'z': z / mag
            }
        
        norm_accel = normalize(accel_data)
        norm_gyro = normalize(gyro_data)
        norm_mag = normalize(mag_data)
        
        # Color by time if timestamps provided
        if timestamps is not None:
            colors = timestamps
            cmap = plt.cm.viridis
        else:
            colors = np.linspace(0, 1, len(norm_accel['x']))
            cmap = plt.cm.plasma
        
        # Plot normalized accelerometer
        self.ax.plot(norm_accel['x'], norm_accel['y'], norm_accel['z'], 
                    color='#4ecdc4', linewidth=1.5, alpha=0.8, label='Accelerometer')
        
        # Plot normalized gyroscope
        self.ax.plot(norm_gyro['x'], norm_gyro['y'], norm_gyro['z'], 
                    color='#45b7d1', linewidth=1.5, alpha=0.8, label='Gyroscope')
        
        # Plot normalized magnetometer
        self.ax.plot(norm_mag['x'], norm_mag['y'], norm_mag['z'], 
                    color='#a55eea', linewidth=1.5, alpha=0.8, label='Magnetometer')
        
        # Show fused vectors (average direction)
        fused_x = (norm_accel['x'] + norm_gyro['x'] + norm_mag['x']) / 3
        fused_y = (norm_accel['y'] + norm_gyro['y'] + norm_mag['y']) / 3
        fused_z = (norm_accel['z'] + norm_gyro['z'] + norm_mag['z']) / 3
        
        self.ax.plot(fused_x, fused_y, fused_z, color='#fd9644', linewidth=2, 
                    alpha=0.9, linestyle='--', label='Fused Direction')
        
        # Set labels
        self.ax.set_xlabel('X (Normalized)', color='white')
        self.ax.set_ylabel('Y (Normalized)', color='white')
        self.ax.set_zlabel('Z (Normalized)', color='white')
        
        # Set limits
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        
        self.ax.legend()
        
        return self.to_base64()


def generate_3d_visualizations(packet) -> Dict[str, str]:
    """
    Generate all available 3D visualizations from RedVox packet.
    
    Args:
        packet: RedVox packet object
    
    Returns:
        Dictionary with visualization names and base64 encoded images
    """
    sensors = packet.get_sensors()
    visualizations = {}
    
    # Accelerometer 3D
    if sensors.has_accelerometer():
        accel = sensors.get_accelerometer()
        accel_data = {
            'x': accel.get_x_samples().get_values(),
            'y': accel.get_y_samples().get_values(),
            'z': accel.get_z_samples().get_values()
        }
        timestamps = accel.get_timestamps().get_timestamps()
        
        visualizer = Accelerometer3DVisualizer()
        visualizations['accelerometer_3d'] = visualizer.visualize(accel_data, timestamps)
    
    # Gyroscope 3D
    if sensors.has_gyroscope():
        gyro = sensors.get_gyroscope()
        gyro_data = {
            'x': gyro.get_x_samples().get_values(),
            'y': gyro.get_y_samples().get_values(),
            'z': gyro.get_z_samples().get_values()
        }
        timestamps = gyro.get_timestamps().get_timestamps()
        
        visualizer = Gyroscope3DVisualizer()
        visualizations['gyroscope_3d'] = visualizer.visualize(gyro_data, timestamps)
    
    # Magnetometer 3D
    if sensors.has_magnetometer():
        mag = sensors.get_magnetometer()
        mag_data = {
            'x': mag.get_x_samples().get_values(),
            'y': mag.get_y_samples().get_values(),
            'z': mag.get_z_samples().get_values()
        }
        timestamps = mag.get_timestamps().get_timestamps()
        
        visualizer = Magnetometer3DVisualizer()
        visualizations['magnetometer_3d'] = visualizer.visualize(mag_data, timestamps)
    
    # Location 3D
    if sensors.has_location():
        location = sensors.get_location()
        location_data = {
            'latitude': location.get_latitude_samples().get_values(),
            'longitude': location.get_longitude_samples().get_values(),
            'altitude': location.get_altitude_samples().get_values() if location.has_altitude() else np.array([])
        }
        timestamps = location.get_timestamps().get_timestamps()
        
        visualizer = Location3DVisualizer()
        visualizations['location_3d'] = visualizer.visualize(location_data, timestamps)
    
    # Multi-sensor fusion 3D
    if (sensors.has_accelerometer() and 
        sensors.has_gyroscope() and 
        sensors.has_magnetometer()):
        
        accel = sensors.get_accelerometer()
        accel_data = {
            'x': accel.get_x_samples().get_values(),
            'y': accel.get_y_samples().get_values(),
            'z': accel.get_z_samples().get_values()
        }
        
        gyro = sensors.get_gyroscope()
        gyro_data = {
            'x': gyro.get_x_samples().get_values(),
            'y': gyro.get_y_samples().get_values(),
            'z': gyro.get_z_samples().get_values()
        }
        
        mag = sensors.get_magnetometer()
        mag_data = {
            'x': mag.get_x_samples().get_values(),
            'y': mag.get_y_samples().get_values(),
            'z': mag.get_z_samples().get_values()
        }
        
        timestamps = accel.get_timestamps().get_timestamps()
        
        visualizer = MultiSensor3DVisualizer()
        visualizations['sensor_fusion_3d'] = visualizer.visualize_sensor_fusion(
            accel_data, gyro_data, mag_data, timestamps
        )
    
    return visualizations
