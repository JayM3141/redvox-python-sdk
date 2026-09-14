"""
Bulk processing utilities for RedVox data analysis.
Provides directory-level analysis and batch processing capabilities.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from redvox.api1000.wrapped_redvox_packet.wrapped_packet import WrappedRedvoxPacketM


class BulkProcessor:
    """Handles bulk processing of RedVox files."""
    
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.results = []
        self.errors = []
        
    def find_rdvxm_files(self, recursive: bool = True) -> List[Path]:
        """Find all .rdvxm files in the input directory."""
        if recursive:
            return list(self.input_dir.rglob("*.rdvxm"))
        else:
            return list(self.input_dir.glob("*.rdvxm"))
    
    def find_rdvxz_files(self, recursive: bool = True) -> List[Path]:
        """Find all .rdvxz files in the input directory."""
        if recursive:
            return list(self.input_dir.rglob("*.rdvxz"))
        else:
            return list(self.input_dir.glob("*.rdvxz"))
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single RedVox file and extract metadata."""
        try:
            packet = WrappedRedvoxPacketM.from_compressed_path(str(file_path))
            
            # Extract hardware information
            station_info = packet.get_station_information()
            sensors_obj = packet.get_sensors()
            
            result = {
                'filename': file_path.name,
                'path': str(file_path),
                'file_size': file_path.stat().st_size,
                'station_id': station_info.get_id(),
                'device_model': f"{station_info.get_make()} {station_info.get_model()}",
                'os_version': f"{station_info.get_os()} {station_info.get_os_version()}",
                'app_version': station_info.get_app_version(),
                'api': 'API 1000/M',
                'available_sensors': [],
                'sensor_counts': {},
            }
            
            # Count available sensors
            sensor_mapping = {
                'accelerometer': sensors_obj.has_accelerometer,
                'ambient_temperature': sensors_obj.has_ambient_temperature,
                'audio': sensors_obj.has_audio,
                'compressed_audio': sensors_obj.has_compressed_audio,
                'gravity': sensors_obj.has_gravity,
                'gyroscope': sensors_obj.has_gyroscope,
                'image': sensors_obj.has_image,
                'light': sensors_obj.has_light,
                'linear_acceleration': sensors_obj.has_linear_acceleration,
                'location': sensors_obj.has_location,
                'magnetometer': sensors_obj.has_magnetometer,
                'orientation': sensors_obj.has_orientation,
                'pressure': sensors_obj.has_pressure,
                'proximity': sensors_obj.has_proximity,
                'relative_humidity': sensors_obj.has_relative_humidity,
                'rotation_vector': sensors_obj.has_rotation_vector,
                'velocity': sensors_obj.has_velocity,
            }
            
            for sensor_name, has_sensor in sensor_mapping.items():
                if has_sensor():
                    result['available_sensors'].append(sensor_name)
                    # Count samples if possible
                    sensor_obj = getattr(sensors_obj, f'get_{sensor_name}', lambda: None)()
                    if sensor_obj:
                        if hasattr(sensor_obj, 'get_samples'):
                            samples = sensor_obj.get_samples()
                            if hasattr(samples, 'get_values'):
                                result['sensor_counts'][sensor_name] = len(samples.get_values())
                        elif hasattr(sensor_obj, 'get_x_samples'):
                            x_samples = sensor_obj.get_x_samples()
                            if hasattr(x_samples, 'get_values'):
                                result['sensor_counts'][sensor_name] = len(x_samples.get_values())
            
            return result
            
        except Exception as e:
            return {
                'filename': file_path.name,
                'path': str(file_path),
                'error': str(e)
            }
    
    def process_directory(self, recursive: bool = True) -> Dict[str, Any]:
        """Process all RedVox files in the directory."""
        rdvxm_files = self.find_rdvxm_files(recursive)
        rdvxz_files = self.find_rdvxz_files(recursive)
        
        all_files = rdvxm_files + rdvxz_files
        
        for file_path in all_files:
            result = self.process_file(file_path)
            if 'error' in result:
                self.errors.append(result)
            else:
                self.results.append(result)
        
        return {
            'total_files': len(all_files),
            'successful': len(self.results),
            'failed': len(self.errors),
            'results': self.results,
            'errors': self.errors
        }
    
    def create_master_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from processing results."""
        return pd.DataFrame(self.results)
    
    def export_to_csv(self, output_path: str):
        """Export results to CSV file."""
        df = self.create_master_dataframe()
        df.to_csv(output_path, index=False)
    
    def export_to_excel(self, output_path: str):
        """Export results to Excel file."""
        df = self.create_master_dataframe()
        df.to_excel(output_path, index=False)
    
    def export_to_parquet(self, output_path: str):
        """Export results to Parquet file."""
        df = self.create_master_dataframe()
        df.to_parquet(output_path, index=False)
