"""
Real-time synchronized multi-track video generation for RedVox sensor data.
Generates scientific videos with synchronized audio, accelerometer, and location tracks.
"""

import os
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io


class TrackGenerator:
    """Base class for generating individual video tracks."""
    
    def __init__(self, sensor_data: Dict, timestamps: np.ndarray):
        self.sensor_data = sensor_data
        self.timestamps = timestamps
        self.duration = timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0
        self.sample_rate = len(timestamps) / self.duration if self.duration > 0 else 1
    
    def generate_frames(self, fps: int = 30) -> List[np.ndarray]:
        """Generate video frames for this track."""
        raise NotImplementedError


class AudioWaveformTrack(TrackGenerator):
    """Generate audio waveform visualization track."""
    
    def __init__(self, audio_samples: np.ndarray, sample_rate: float):
        self.audio_samples = audio_samples
        self.sample_rate = sample_rate
        self.duration = len(audio_samples) / sample_rate
        self.timestamps = np.linspace(0, self.duration, len(audio_samples))
    
    def generate_frames(self, fps: int = 30) -> List[np.ndarray]:
        """Generate waveform frames."""
        frames = []
        total_frames = int(self.duration * fps)
        
        fig, ax = plt.subplots(figsize=(16, 4), facecolor='#1e1e2e')
        ax.set_facecolor('#1e1e2e')
        ax.set_title('Audio Waveform', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        for frame_idx in range(total_frames):
            time_point = frame_idx / fps
            window_size = 0.1  # 100ms window
            start_idx = int((time_point - window_size/2) * self.sample_rate)
            end_idx = int((time_point + window_size/2) * self.sample_rate)
            
            start_idx = max(0, start_idx)
            end_idx = min(len(self.audio_samples), end_idx)
            
            ax.clear()
            ax.set_facecolor('#1e1e2e')
            ax.set_title(f'Audio Waveform - {time_point:.2f}s', color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
            if end_idx > start_idx:
                window_samples = self.audio_samples[start_idx:end_idx]
                window_time = np.linspace(time_point - window_size/2, time_point + window_size/2, len(window_samples))
                ax.plot(window_time, window_samples, color='#00ff88', linewidth=1)
                ax.set_ylim(np.min(self.audio_samples), np.max(self.audio_samples))
            
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            frames.append(buf)
        
        plt.close(fig)
        return frames


class AccelerometerAnimationTrack(TrackGenerator):
    """Generate 3D accelerometer visualization track."""
    
    def __init__(self, accel_data: Dict[str, np.ndarray], timestamps: np.ndarray):
        super().__init__(accel_data, timestamps)
        self.x = accel_data['x']
        self.y = accel_data['y']
        self.z = accel_data['z']
    
    def generate_frames(self, fps: int = 30) -> List[np.ndarray]:
        """Generate 3D accelerometer frames."""
        frames = []
        total_frames = int(self.duration * fps)
        
        fig = plt.figure(figsize=(8, 8), facecolor='#1e1e2e')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#1e1e2e')
        
        for frame_idx in range(total_frames):
            time_point = frame_idx / fps
            idx = int(time_point * self.sample_rate)
            idx = min(idx, len(self.x) - 1)
            
            ax.clear()
            ax.set_facecolor('#1e1e2e')
            ax.set_title(f'Accelerometer - {time_point:.2f}s', color='white')
            ax.tick_params(colors='white')
            
            # Draw 3D vector
            x_val = self.x[idx]
            y_val = self.y[idx]
            z_val = self.z[idx]
            
            ax.quiver(0, 0, 0, x_val, y_val, z_val, color='#ff6b6b', linewidth=2)
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            ax.set_zlim(-20, 20)
            ax.set_xlabel('X', color='white')
            ax.set_ylabel('Y', color='white')
            ax.set_zlabel('Z', color='white')
            
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            frames.append(buf)
        
        plt.close(fig)
        return frames


class LocationPathTrack(TrackGenerator):
    """Generate location path animation track."""
    
    def __init__(self, location_data: Dict[str, np.ndarray], timestamps: np.ndarray):
        super().__init__(location_data, timestamps)
        self.latitude = location_data['latitude']
        self.longitude = location_data['longitude']
        self.altitude = location_data.get('altitude', np.array([]))
    
    def generate_frames(self, fps: int = 30) -> List[np.ndarray]:
        """Generate location path frames."""
        frames = []
        total_frames = int(self.duration * fps)
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1e1e2e')
        ax.set_facecolor('#1e1e2e')
        
        for frame_idx in range(total_frames):
            time_point = frame_idx / fps
            idx = int(time_point * self.sample_rate)
            idx = min(idx, len(self.latitude) - 1)
            
            ax.clear()
            ax.set_facecolor('#1e1e2e')
            ax.set_title(f'Location Path - {time_point:.2f}s', color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
            # Plot path up to current point
            ax.plot(self.longitude[:idx], self.latitude[:idx], color='#4ecdc4', linewidth=2, alpha=0.7)
            ax.scatter(self.longitude[idx], self.latitude[idx], color='#ff6b6b', s=100, zorder=5)
            
            ax.set_xlabel('Longitude', color='white')
            ax.set_ylabel('Latitude', color='white')
            ax.grid(True, alpha=0.3)
            
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            frames.append(buf)
        
        plt.close(fig)
        return frames


class SynchronizedVideoGenerator:
    """Generate synchronized multi-track videos from RedVox sensor data."""
    
    def __init__(self, output_path: str, fps: int = 30):
        self.output_path = output_path
        self.fps = fps
        self.tracks = []
        self.temp_dir = tempfile.mkdtemp()
    
    def add_audio_track(self, audio_samples: np.ndarray, sample_rate: float):
        """Add audio waveform track."""
        track = AudioWaveformTrack(audio_samples, sample_rate)
        self.tracks.append(('audio', track))
    
    def add_accelerometer_track(self, accel_data: Dict[str, np.ndarray], timestamps: np.ndarray):
        """Add accelerometer 3D animation track."""
        track = AccelerometerAnimationTrack(accel_data, timestamps)
        self.tracks.append(('accelerometer', track))
    
    def add_location_track(self, location_data: Dict[str, np.ndarray], timestamps: np.ndarray):
        """Add location path animation track."""
        track = LocationPathTrack(location_data, timestamps)
        self.tracks.append(('location', track))
    
    def generate_track_video(self, track_name: str, track: TrackGenerator) -> str:
        """Generate video for a single track."""
        frames = track.generate_frames(self.fps)
        
        if not frames:
            return None
        
        # Save frames as temporary video file
        temp_video = os.path.join(self.temp_dir, f"{track_name}.mp4")
        
        # Use FFmpeg to create video from frames
        height, width = frames[0].shape[:2]
        
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            temp_video
        ]
        
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
        for frame in frames:
            process.stdin.write(frame.tobytes())
        
        process.stdin.close()
        process.wait()
        
        return temp_video
    
    def generate_synchronized_video(self):
        """Generate final synchronized video with all tracks."""
        if not self.tracks:
            raise ValueError("No tracks added")
        
        # Generate individual track videos
        track_videos = []
        for track_name, track in self.tracks:
            video_path = self.generate_track_video(track_name, track)
            if video_path:
                track_videos.append(video_path)
        
        if not track_videos:
            raise ValueError("No track videos generated")
        
        # Combine tracks using FFmpeg
        if len(track_videos) == 1:
            # Single track, just rename
            import shutil
            shutil.move(track_videos[0], self.output_path)
        else:
            # Multiple tracks, arrange in grid
            filter_complex = ""
            
            if len(track_videos) == 2:
                filter_complex = f"[0:v][1:v]hstack=inputs=2[v]"
            elif len(track_videos) == 3:
                filter_complex = f"[0:v][1:v]hstack=inputs=2[top];[2:v]nullsink;[top][2:v]vstack=inputs=2[v]"
            elif len(track_videos) == 4:
                filter_complex = f"[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]"
            else:
                # More than 4 tracks, arrange in grid
                grid_size = int(np.ceil(np.sqrt(len(track_videos))))
                filter_complex = f"[0:v]scale=iw/2:ih/2[s0]"
                for i in range(1, len(track_videos)):
                    filter_complex += f";[{i}:v]scale=iw/2:ih/2[s{i}]"
                filter_complex += f";"
                for i in range(len(track_videos)):
                    filter_complex += f"[s{i}]"
                filter_complex += f"xstack=inputs={len(track_videos)}:layout=0_0|w0_0|0_h0|w0_h0[v]"
            
            cmd = [
                'ffmpeg',
                '-y',
            ]
            
            for video in track_videos:
                cmd.extend(['-i', video])
            
            cmd.extend([
                '-filter_complex', filter_complex,
                '-map', '[v]',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                self.output_path
            ])
            
            subprocess.run(cmd, check=True)
        
        # Cleanup
        for video in track_videos:
            if os.path.exists(video):
                os.remove(video)
        
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
        
        return self.output_path


def generate_scientific_video(packet, output_path: str, include_audio: bool = True,
                              include_accelerometer: bool = True, include_location: bool = True,
                              fps: int = 30) -> str:
    """
    Generate a scientific video from RedVox packet data.
    
    Args:
        packet: RedVox packet object
        output_path: Output video file path
        include_audio: Include audio waveform track
        include_accelerometer: Include accelerometer 3D track
        include_location: Include location path track
        fps: Frames per second for video
    
    Returns:
        Path to generated video file
    """
    sensors = packet.get_sensors()
    generator = SynchronizedVideoGenerator(output_path, fps)
    
    # Add audio track
    if include_audio and sensors.has_audio():
        audio = sensors.get_audio()
        audio_samples = audio.get_samples().get_values()
        sample_rate = audio.get_sample_rate()
        generator.add_audio_track(audio_samples, sample_rate)
    
    # Add accelerometer track
    if include_accelerometer and sensors.has_accelerometer():
        accel = sensors.get_accelerometer()
        accel_data = {
            'x': accel.get_x_samples().get_values(),
            'y': accel.get_y_samples().get_values(),
            'z': accel.get_z_samples().get_values()
        }
        timestamps = accel.get_timestamps().get_timestamps()
        generator.add_accelerometer_track(accel_data, timestamps)
    
    # Add location track
    if include_location and sensors.has_location():
        location = sensors.get_location()
        location_data = {
            'latitude': location.get_latitude_samples().get_values(),
            'longitude': location.get_longitude_samples().get_values(),
            'altitude': location.get_altitude_samples().get_values() if location.has_altitude() else np.array([])
        }
        timestamps = location.get_timestamps().get_timestamps()
        generator.add_location_track(location_data, timestamps)
    
    return generator.generate_synchronized_video()
