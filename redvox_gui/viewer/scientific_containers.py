"""
Scientific container support for RedVox data.
Provides HDF5 and NetCDF export for scientific data storage and analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


class HDF5Exporter:
    """Export RedVox data to HDF5 format for scientific analysis."""
    
    def __init__(self):
        self.h5py_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if h5py is available."""
        try:
            import h5py
            self.h5py_available = True
            self.h5py = h5py
        except ImportError:
            self.h5py_available = False
    
    def export_to_hdf5(self, packet, output_path: str, 
                      compression: str = 'gzip', 
                      compression_level: int = 4) -> bool:
        """
        Export RedVox packet to HDF5 format.
        
        Args:
            packet: RedVox packet object
            output_path: Output HDF5 file path
            compression: Compression method ('gzip', 'lzf', 'szip')
            compression_level: Compression level (0-9 for gzip)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.h5py_available:
            print("h5py not installed. Install with: pip install h5py")
            return False
        
        try:
            with self.h5py.File(output_path, 'w') as h5file:
                # Create metadata group
                metadata_group = h5file.create_group('metadata')
                
                # Add packet metadata
                station_info = packet.get_station_information()
                metadata_group.attrs['station_id'] = station_info.get_id()
                metadata_group.attrs['uuid'] = station_info.get_uuid()
                metadata_group.attrs['make'] = station_info.get_make()
                metadata_group.attrs['model'] = station_info.get_model()
                metadata_group.attrs['os'] = station_info.get_os()
                metadata_group.attrs['os_version'] = station_info.get_os_version()
                metadata_group.attrs['app_version'] = station_info.get_app_version()
                
                timing_info = packet.get_timing_information()
                metadata_group.attrs['packet_start'] = timing_info.get_packet_start_mach()
                metadata_group.attrs['packet_end'] = timing_info.get_packet_end_mach()
                metadata_group.attrs['export_timestamp'] = datetime.now().isoformat()
                
                # Create sensors group
                sensors = packet.get_sensors()
                sensors_group = h5file.create_group('sensors')
                
                # Export audio
                if sensors.has_audio():
                    audio = sensors.get_audio()
                    audio_group = sensors_group.create_group('audio')
                    
                    audio_samples = audio.get_samples().get_values()
                    audio_group.create_dataset(
                        'samples',
                        data=audio_samples,
                        compression=compression,
                        compression_opts=compression_level
                    )
                    audio_group.attrs['sample_rate'] = audio.get_sample_rate()
                    audio_group.attrs['sample_count'] = len(audio_samples)
                    audio_group.attrs['unit'] = 'NORMALIZED_COUNTS'
                
                # Export accelerometer
                if sensors.has_accelerometer():
                    accel = sensors.get_accelerometer()
                    accel_group = sensors_group.create_group('accelerometer')
                    
                    for axis in ['x', 'y', 'z']:
                        axis_samples = getattr(accel, f'get_{axis}_samples')().get_values()
                        accel_group.create_dataset(
                            axis,
                            data=axis_samples,
                            compression=compression,
                            compression_opts=compression_level
                        )
                    
                    timestamps = accel.get_timestamps().get_timestamps()
                    accel_group.create_dataset(
                        'timestamps',
                        data=timestamps,
                        compression=compression,
                        compression_opts=compression_level
                    )
                    accel_group.attrs['sample_rate'] = accel.get_sample_rate()
                    accel_group.attrs['unit'] = 'METERS_PER_SECOND_SQUARED'
                
                # Export gyroscope
                if sensors.has_gyroscope():
                    gyro = sensors.get_gyroscope()
                    gyro_group = sensors_group.create_group('gyroscope')
                    
                    for axis in ['x', 'y', 'z']:
                        axis_samples = getattr(gyro, f'get_{axis}_samples')().get_values()
                        gyro_group.create_dataset(
                            axis,
                            data=axis_samples,
                            compression=compression,
                            compression_opts=compression_level
                        )
                    
                    timestamps = gyro.get_timestamps().get_timestamps()
                    gyro_group.create_dataset(
                        'timestamps',
                        data=timestamps,
                        compression=compression,
                        compression_opts=compression_level
                    )
                    gyro_group.attrs['sample_rate'] = gyro.get_sample_rate()
                    gyro_group.attrs['unit'] = 'RADIANS_PER_SECOND'
                
                # Export magnetometer
                if sensors.has_magnetometer():
                    mag = sensors.get_magnetometer()
                    mag_group = sensors_group.create_group('magnetometer')
                    
                    for axis in ['x', 'y', 'z']:
                        axis_samples = getattr(mag, f'get_{axis}_samples')().get_values()
                        mag_group.create_dataset(
                            axis,
                            data=axis_samples,
                            compression=compression,
                            compression_opts=compression_level
                        )
                    
                    timestamps = mag.get_timestamps().get_timestamps()
                    mag_group.create_dataset(
                        'timestamps',
                        data=timestamps,
                        compression=compression,
                        compression_opts=compression_level
                    )
                    mag_group.attrs['sample_rate'] = mag.get_sample_rate()
                    mag_group.attrs['unit'] = 'MICROTESLA'
                
                # Export pressure
                if sensors.has_pressure():
                    pressure = sensors.get_pressure()
                    pressure_group = sensors_group.create_group('pressure')
                    
                    pressure_samples = pressure.get_samples().get_values()
                    pressure_group.create_dataset(
                        'samples',
                        data=pressure_samples,
                        compression=compression,
                        compression_opts=compression_level
                    )
                    
                    timestamps = pressure.get_timestamps().get_timestamps()
                    pressure_group.create_dataset(
                        'timestamps',
                        data=timestamps,
                        compression=compression,
                        compression_opts=compression_level
                    )
                    pressure_group.attrs['sample_rate'] = pressure.get_sample_rate()
                    pressure_group.attrs['unit'] = 'KILOPASCAL'
                
                # Export light
                if sensors.has_light():
                    light = sensors.get_light()
                    light_group = sensors_group.create_group('light')
                    
                    light_samples = light.get_samples().get_values()
                    light_group.create_dataset(
                        'samples',
                        data=light_samples,
                        compression=compression,
                        compression_opts=compression_level
                    )
                    
                    timestamps = light.get_timestamps().get_timestamps()
                    light_group.create_dataset(
                        'timestamps',
                        data=timestamps,
                        compression=compression,
                        compression_opts=compression_level
                    )
                    light_group.attrs['sample_rate'] = light.get_sample_rate()
                    light_group.attrs['unit'] = 'LUX'
                
                # Export location
                if sensors.has_location():
                    location = sensors.get_location()
                    location_group = sensors_group.create_group('location')
                    
                    location_group.create_dataset(
                        'latitude',
                        data=location.get_latitude_samples().get_values(),
                        compression=compression,
                        compression_opts=compression_level
                    )
                    location_group.create_dataset(
                        'longitude',
                        data=location.get_longitude_samples().get_values(),
                        compression=compression,
                        compression_opts=compression_level
                    )
                    
                    if location.has_altitude():
                        location_group.create_dataset(
                            'altitude',
                            data=location.get_altitude_samples().get_values(),
                            compression=compression,
                            compression_opts=compression_level
                        )
                    
                    if location.has_speed():
                        location_group.create_dataset(
                            'speed',
                            data=location.get_speed_samples().get_values(),
                            compression=compression,
                            compression_opts=compression_level
                        )
                    
                    timestamps = location.get_timestamps().get_timestamps()
                    location_group.create_dataset(
                        'timestamps',
                        data=timestamps,
                        compression=compression,
                        compression_opts=compression_level
                    )
                    location_group.attrs['sample_rate'] = location.get_sample_rate()
                    location_group.attrs['latitude_unit'] = 'DECIMAL_DEGREES'
                    location_group.attrs['longitude_unit'] = 'DECIMAL_DEGREES'
                    location_group.attrs['altitude_unit'] = 'METERS'
                
                # Add description
                h5file.attrs['description'] = 'RedVox scientific data export'
                h5file.attrs['format_version'] = '1.0'
                h5file.attrs['creator'] = 'RedVox Python SDK'
                
            return True
            
        except Exception as e:
            print(f"Error exporting to HDF5: {e}")
            return False


class NetCDFExporter:
    """Export RedVox data to NetCDF format for scientific analysis."""
    
    def __init__(self):
        self.netcdf_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if netCDF4 is available."""
        try:
            import netCDF4
            self.netcdf_available = True
            self.netCDF4 = netCDF4
        except ImportError:
            self.netcdf_available = False
    
    def export_to_netcdf(self, packet, output_path: str) -> bool:
        """
        Export RedVox packet to NetCDF format.
        
        Args:
            packet: RedVox packet object
            output_path: Output NetCDF file path
        
        Returns:
            True if successful, False otherwise
        """
        if not self.netcdf_available:
            print("netCDF4 not installed. Install with: pip install netCDF4")
            return False
        
        try:
            with self.netCDF4.Dataset(output_path, 'w', format='NETCDF4') as ncfile:
                # Add global attributes
                station_info = packet.get_station_information()
                ncfile.title = 'RedVox Scientific Data Export'
                ncfile.institution = 'RedVox'
                ncfile.source = 'RedVox Python SDK'
                ncfile.history = f'Created {datetime.now().isoformat()}'
                ncfile.station_id = station_info.get_id()
                ncfile.uuid = station_info.get_uuid()
                ncfile.device_make = station_info.get_make()
                ncfile.device_model = station_info.get_model()
                
                # Get sensors
                sensors = packet.get_sensors()
                
                # Export audio
                if sensors.has_audio():
                    audio = sensors.get_audio()
                    audio_samples = audio.get_samples().get_values()
                    
                    ncfile.createDimension('audio_time', len(audio_samples))
                    
                    audio_var = ncfile.createVariable(
                        'audio_samples',
                        'f4',
                        ('audio_time',),
                        zlib=True,
                        complevel=4
                    )
                    audio_var[:] = audio_samples
                    audio_var.units = 'NORMALIZED_COUNTS'
                    audio_var.sample_rate = audio.get_sample_rate()
                    audio_var.long_name = 'Audio samples'
                
                # Export accelerometer
                if sensors.has_accelerometer():
                    accel = sensors.get_accelerometer()
                    accel_timestamps = accel.get_timestamps().get_timestamps()
                    
                    ncfile.createDimension('accel_time', len(accel_timestamps))
                    
                    time_var = ncfile.createVariable(
                        'accel_time',
                        'f8',
                        ('accel_time',)
                    )
                    time_var[:] = accel_timestamps
                    time_var.units = 'microseconds since epoch'
                    time_var.long_name = 'Accelerometer timestamps'
                    
                    for axis in ['x', 'y', 'z']:
                        axis_samples = getattr(accel, f'get_{axis}_samples')().get_values()
                        axis_var = ncfile.createVariable(
                            f'accel_{axis}',
                            'f4',
                            ('accel_time',),
                            zlib=True,
                            complevel=4
                        )
                        axis_var[:] = axis_samples
                        axis_var.units = 'METERS_PER_SECOND_SQUARED'
                        axis_var.long_name = f'Accelerometer {axis.upper()}'
                
                # Export gyroscope
                if sensors.has_gyroscope():
                    gyro = sensors.get_gyroscope()
                    gyro_timestamps = gyro.get_timestamps().get_timestamps()
                    
                    ncfile.createDimension('gyro_time', len(gyro_timestamps))
                    
                    time_var = ncfile.createVariable(
                        'gyro_time',
                        'f8',
                        ('gyro_time',)
                    )
                    time_var[:] = gyro_timestamps
                    time_var.units = 'microseconds since epoch'
                    time_var.long_name = 'Gyroscope timestamps'
                    
                    for axis in ['x', 'y', 'z']:
                        axis_samples = getattr(gyro, f'get_{axis}_samples')().get_values()
                        axis_var = ncfile.createVariable(
                            f'gyro_{axis}',
                            'f4',
                            ('gyro_time',),
                            zlib=True,
                            complevel=4
                        )
                        axis_var[:] = axis_samples
                        axis_var.units = 'RADIANS_PER_SECOND'
                        axis_var.long_name = f'Gyroscope {axis.upper()}'
                
                # Export location
                if sensors.has_location():
                    location = sensors.get_location()
                    location_timestamps = location.get_timestamps().get_timestamps()
                    
                    ncfile.createDimension('location_time', len(location_timestamps))
                    
                    time_var = ncfile.createVariable(
                        'location_time',
                        'f8',
                        ('location_time',)
                    )
                    time_var[:] = location_timestamps
                    time_var.units = 'microseconds since epoch'
                    time_var.long_name = 'Location timestamps'
                    
                    lat_var = ncfile.createVariable(
                        'latitude',
                        'f4',
                        ('location_time',),
                        zlib=True,
                        complevel=4
                    )
                    lat_var[:] = location.get_latitude_samples().get_values()
                    lat_var.units = 'DECIMAL_DEGREES'
                    lat_var.long_name = 'Latitude'
                    
                    lon_var = ncfile.createVariable(
                        'longitude',
                        'f4',
                        ('location_time',),
                        zlib=True,
                        complevel=4
                    )
                    lon_var[:] = location.get_longitude_samples().get_values()
                    lon_var.units = 'DECIMAL_DEGREES'
                    lon_var.long_name = 'Longitude'
                    
                    if location.has_altitude():
                        alt_var = ncfile.createVariable(
                            'altitude',
                            'f4',
                            ('location_time',),
                            zlib=True,
                            complevel=4
                        )
                        alt_var[:] = location.get_altitude_samples().get_values()
                        alt_var.units = 'METERS'
                        alt_var.long_name = 'Altitude'
                
            return True
            
        except Exception as e:
            print(f"Error exporting to NetCDF: {e}")
            return False


def export_to_scientific_container(packet, output_path: str, 
                                    format: str = 'hdf5') -> bool:
    """
    Export RedVox packet to scientific container format.
    
    Args:
        packet: RedVox packet object
        output_path: Output file path
        format: Container format ('hdf5' or 'netcdf')
    
    Returns:
        True if successful, False otherwise
    """
    if format == 'hdf5':
        exporter = HDF5Exporter()
        return exporter.export_to_hdf5(packet, output_path)
    elif format == 'netcdf':
        exporter = NetCDFExporter()
        return exporter.export_to_netcdf(packet, output_path)
    else:
        print(f"Unknown format: {format}")
        return False
