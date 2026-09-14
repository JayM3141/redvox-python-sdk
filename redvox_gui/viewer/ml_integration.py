"""
Machine Learning integration for RedVox sensor data classification.
Provides YAMNet audio classification and extends to multi-sensor classification.
Includes event classification for all sensors (accelerometer, gyroscope, magnetometer, light, pressure, location).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import signal
from scipy.stats import zscore
warnings.filterwarnings('ignore')


class AudioClassifier:
    """Base class for audio classification models."""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
    
    def load_model(self):
        """Load the ML model. Override in subclasses."""
        raise NotImplementedError
    
    def classify_audio(self, audio_samples: np.ndarray, sample_rate: float) -> Dict[str, float]:
        """Classify audio samples. Override in subclasses."""
        raise NotImplementedError


class YAMNetClassifier(AudioClassifier):
    """YAMNet audio classification model integration."""
    
    def __init__(self):
        super().__init__()
        self.class_names = [
            "Speech", "Babbling", "Speech noise", "Shout", "Cough", "Sneeze",
            "Breathing", "Whistling", "Clapping", "Finger snapping", "Knocking",
            "Keys jingling", "Mouse clicking", "Typing", "Keyboard typing", "Writing",
            "Page turning", "Footsteps", "Walking/running", "Door opening/closing",
            "Drawer open/close", "Hand washing", "Dishes", "Frying", "Blender",
            "Running water", "Vacuum cleaner", "Alarm/bell", "Ringing telephone",
            "Doorbell", "Alarm clock", "Siren", "Smoke alarm", "Fire alarm",
            "Emergency vehicle", "Car horn", "Train whistle", "Boat/ship whistle",
            "Police siren", "Civil defense siren", "Drilling", "Hammering",
            "Chainsaw", "Lawn mower", "Leaf blower", "Power tool", "Sawing",
            "Sandpaper", "Power drill", "Jackhammer", "Explosion", "Gunshot",
            "Firecrackers", "Fireworks", "Bang", "Fire", "Rain", "Thunder",
            "Wind", "Water flowing", "Ocean waves", "Rain drops", "Stream",
            "Waterfall", "Birds singing", "Birds chirping", "Dog barking",
            "Cat meowing", "Dog whining", "Bird flapping", "Insect buzzing",
            "Insect chirping", "Frog croaking", "Insect flight", "Crickickets",
            "Mosquito buzzing", "Baby crying", "Baby laughing", "Child speech",
            "Child singing", "Child laughing", "Child crying", "Baby sneezing",
            "Baby coughing", "Child coughing", "Baby snoring", "Child snoring",
            "Baby cooing", "Baby wheezing", "Child giggling", "Child sobbing",
            "Adult speech", "Adult laughing", "Adult coughing", "Adult sneezing",
            "Adult snoring", "Adult wheezing", "Adult singing", "Female speech",
            "Male speech", "Female singing", "Male singing", "Female laughing",
            "Male laughing", "Female coughing", "Male coughing", "Female sneezing",
            "Male sneezing", "Breathing noise", "Heartbeat", "Pulse", "Snoring",
            "Asthma", "Wheezing", "Choking", "Gagging", "Hiccup", "Burping",
            "Yawn", "Sneeze", "Cough", "Laugh", "Cry", "Groan", "Moan",
            "Whimper", "Sob", "Whine", "Scream", "Yell", "Shout", "Cheer",
            "Applause", "Clapping", "Cheering", "Booing", "Silence"
        ]
    
    def load_model(self):
        """Load YAMNet model from TensorFlow Hub."""
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            
            # Load YAMNet model from TensorFlow Hub
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            self.model_loaded = True
            return True
        except ImportError:
            print("TensorFlow not installed. Install with: pip install tensorflow tensorflow-hub")
            return False
        except Exception as e:
            print(f"Error loading YAMNet model: {e}")
            return False
    
    def classify_audio(self, audio_samples: np.ndarray, sample_rate: float) -> Dict[str, float]:
        """Classify audio using YAMNet."""
        if not self.model_loaded:
            if not self.load_model():
                return {"error": "Model not loaded"}
        
        try:
            import tensorflow as tf
            
            # Convert audio to float32 and normalize
            audio_float32 = audio_samples.astype(np.float32)
            
            # Resample to 16kHz if needed (YAMNet expects 16kHz)
            if sample_rate != 16000:
                from scipy import signal
                num_samples = int(len(audio_float32) * 16000 / sample_rate)
                audio_float32 = signal.resample(audio_float32, num_samples)
            
            # Ensure audio is mono
            if len(audio_float32.shape) > 1:
                audio_float32 = np.mean(audio_float32, axis=1)
            
            # Normalize to [-1, 1]
            if np.max(np.abs(audio_float32)) > 0:
                audio_float32 = audio_float32 / np.max(np.abs(audio_float32))
            
            # Run inference
            waveform = tf.constant(audio_float32, dtype=tf.float32)
            scores, embeddings, spectrogram = self.model(waveform)
            
            # Get top predictions
            scores_np = scores.numpy()
            mean_scores = np.mean(scores_np, axis=0)
            
            # Return top 5 classifications
            top_indices = np.argsort(mean_scores)[-5:][::-1]
            top_classifications = {
                self.class_names[i]: float(mean_scores[i]) 
                for i in top_indices
            }
            
            return top_classifications
            
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}


class MultiSensorClassifier:
    """Multi-sensor classification for accelerometer, gyroscope, and other sensors."""
    
    def __init__(self):
        self.audio_classifier = YAMNetClassifier()
        self.motion_classifier = None
        self.environmental_classifier = None
    
    def classify_audio(self, audio_samples: np.ndarray, sample_rate: float) -> Dict[str, float]:
        """Classify audio data."""
        return self.audio_classifier.classify_audio(audio_samples, sample_rate)
    
    def classify_motion(self, accelerometer_data: Dict[str, np.ndarray], 
                      gyroscope_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Classify motion patterns from accelerometer and gyroscope data."""
        # This would use a CNN or other ML model for motion classification
        # For now, return placeholder
        return {
            "walking": 0.0,
            "running": 0.0,
            "stationary": 0.0,
            "error": "Motion classification not yet implemented"
        }
    
    def classify_environmental(self, pressure: np.ndarray, temperature: np.ndarray,
                             humidity: np.ndarray) -> Dict[str, float]:
        """Classify environmental conditions."""
        # This would use Random Forest or other ML model for environmental classification
        # For now, return placeholder
        return {
            "indoor": 0.0,
            "outdoor": 0.0,
            "error": "Environmental classification not yet implemented"
        }
    
    def classify_all_sensors(self, sensor_data: Dict) -> Dict[str, Dict[str, float]]:
        """Classify all available sensor data."""
        results = {}
        
        # Audio classification
        if 'audio' in sensor_data:
            audio_data = sensor_data['audio']
            results['audio'] = self.classify_audio(
                audio_data['samples'], 
                audio_data['sample_rate']
            )
        
        # Motion classification
        if 'accelerometer' in sensor_data and 'gyroscope' in sensor_data:
            results['motion'] = self.classify_motion(
                sensor_data['accelerometer'],
                sensor_data['gyroscope']
            )
        
        # Environmental classification
        if 'pressure' in sensor_data or 'temperature' in sensor_data:
            env_data = {
                'pressure': sensor_data.get('pressure', {}).get('samples', np.array([])),
                'temperature': sensor_data.get('temperature', {}).get('samples', np.array([])),
                'humidity': sensor_data.get('humidity', {}).get('samples', np.array([]))
            }
            results['environmental'] = self.classify_environmental(**env_data)
        
        return results


def extract_sensor_data_for_ml(packet) -> Dict:
    """Extract sensor data from RedVox packet for ML classification."""
    sensors = packet.get_sensors()
    sensor_data = {}
    
    # Extract audio data
    if sensors.has_audio():
        audio = sensors.get_audio()
        sensor_data['audio'] = {
            'samples': audio.get_samples().get_values(),
            'sample_rate': audio.get_sample_rate()
        }
    
    # Extract accelerometer data
    if sensors.has_accelerometer():
        accel = sensors.get_accelerometer()
        sensor_data['accelerometer'] = {
            'x': accel.get_x_samples().get_values(),
            'y': accel.get_y_samples().get_values(),
            'z': accel.get_z_samples().get_values(),
            'timestamps': accel.get_timestamps().get_timestamps()
        }
    
    # Extract gyroscope data
    if sensors.has_gyroscope():
        gyro = sensors.get_gyroscope()
        sensor_data['gyroscope'] = {
            'x': gyro.get_x_samples().get_values(),
            'y': gyro.get_y_samples().get_values(),
            'z': gyro.get_z_samples().get_values(),
            'timestamps': gyro.get_timestamps().get_timestamps()
        }
    
    # Extract pressure data
    if sensors.has_pressure():
        pressure = sensors.get_pressure()
        sensor_data['pressure'] = {
            'samples': pressure.get_samples().get_values(),
            'timestamps': pressure.get_timestamps().get_timestamps()
        }
    
    # Extract temperature data
    if sensors.has_ambient_temperature():
        temp = sensors.get_ambient_temperature()
        sensor_data['temperature'] = {
            'samples': temp.get_samples().get_values(),
            'timestamps': temp.get_timestamps().get_timestamps()
        }
    
    # Extract humidity data
    if sensors.has_relative_humidity():
        humidity = sensors.get_relative_humidity()
        sensor_data['humidity'] = {
            'samples': humidity.get_samples().get_values(),
            'timestamps': humidity.get_timestamps().get_timestamps()
        }
    
    # Extract magnetometer data
    if sensors.has_magnetometer():
        mag = sensors.get_magnetometer()
        sensor_data['magnetometer'] = {
            'x': mag.get_x_samples().get_values(),
            'y': mag.get_y_samples().get_values(),
            'z': mag.get_z_samples().get_values(),
            'timestamps': mag.get_timestamps().get_timestamps()
        }
    
    # Extract light data
    if sensors.has_light():
        light = sensors.get_light()
        sensor_data['light'] = {
            'samples': light.get_samples().get_values(),
            'timestamps': light.get_timestamps().get_timestamps()
        }
    
    # Extract location data
    if sensors.has_location():
        location = sensors.get_location()
        sensor_data['location'] = {
            'latitude': location.get_latitude_samples().get_values(),
            'longitude': location.get_longitude_samples().get_values(),
            'altitude': location.get_altitude_samples().get_values() if location.has_altitude() else np.array([]),
            'speed': location.get_speed_samples().get_values() if location.has_speed() else np.array([]),
            'timestamps': location.get_timestamps().get_timestamps()
        }
    
    return sensor_data


class SensorEventClassifier:
    """Event classification for all sensor types with similar structure to audio events."""
    
    def __init__(self):
        self.accelerometer_events = [
            "Stationary", "Walking", "Running", "Jumping", "Falling",
            "Shaking", "Tapping", "Rotating", "Vehicle motion", "Vibration",
            "Sudden acceleration", "Sudden deceleration", "Tilt change",
            "Free fall", "Impact", "Micro-movement", "Periodic motion"
        ]
        
        self.gyroscope_events = [
            "Stationary", "Slow rotation", "Fast rotation", "Oscillation",
            "Shaking", "Tilting", "Turning", "Spinning", "Twisting",
            "Stable orientation", "Orientation change", "Continuous rotation",
            "Intermittent rotation", "Axis-specific rotation", "Multi-axis motion"
        ]
        
        self.magnetometer_events = [
            "Stable field", "Field change", "Magnetic interference",
            "Orientation change", "Near metal", "Far from metal",
            "North-facing", "South-facing", "East-facing", "West-facing",
            "Field strength increase", "Field strength decrease",
            "Compass deviation", "Calibration needed", "Normal variation"
        ]
        
        self.light_events = [
            "Dark", "Dim", "Normal indoor", "Bright indoor", "Outdoor shade",
            "Outdoor direct", "Flash", "Flicker", "Gradual increase",
            "Gradual decrease", "Sudden increase", "Sudden decrease",
            "Stable", "Variable", "Light source change"
        ]
        
        self.pressure_events = [
            "Stable", "Increasing", "Decreasing", "Rapid increase",
            "Rapid decrease", "Altitude change", "Weather pressure change",
            "Door opening", "Door closing", "Elevator use", "HVAC cycle",
            "Normal variation", "Extreme low", "Extreme high", "Pressure spike"
        ]
        
        self.location_events = [
            "Stationary", "Walking", "Driving", "Cycling", "Running",
            "Indoor", "Outdoor", "GPS signal lost", "GPS signal acquired",
            "Entering building", "Leaving building", "Elevator motion",
            "Altitude change", "Direction change", "Speed change",
            "Route deviation", "Location jump", "Stable GPS", "Drifting GPS"
        ]
    
    def detect_peaks(self, data: np.ndarray, threshold: float = 2.0) -> int:
        """Detect peaks in sensor data using z-score threshold."""
        if len(data) == 0:
            return 0
        try:
            z_scores = zscore(data)
            peaks = np.sum(np.abs(z_scores) > threshold)
            return int(peaks)
        except:
            return 0
    
    def detect_frequency_content(self, data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        """Analyze frequency content of sensor data."""
        if len(data) < 2:
            return {"dominant_freq": 0.0, "power": 0.0}
        
        try:
            freqs, psd = signal.welch(data, fs=sample_rate, nperseg=min(256, len(data)))
            dominant_freq = freqs[np.argmax(psd)]
            power = np.sum(psd)
            return {"dominant_freq": float(dominant_freq), "power": float(power)}
        except:
            return {"dominant_freq": 0.0, "power": 0.0}
    
    def classify_accelerometer_events(self, accel_data: Dict[str, np.ndarray], 
                                      sample_rate: float) -> Dict[str, float]:
        """Classify accelerometer events."""
        if not accel_data or 'x' not in accel_data:
            return {"error": "No accelerometer data"}
        
        x = accel_data['x']
        y = accel_data['y']
        z = accel_data['z']
        
        # Calculate magnitude
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        # Detect statistical features
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        peaks = self.detect_peaks(magnitude)
        freq_info = self.detect_frequency_content(magnitude, sample_rate)
        
        # Event classification based on features
        scores = {}
        
        # Stationary vs motion
        if std_mag < 0.1:
            scores["Stationary"] = 0.9
            scores["Micro-movement"] = 0.5
        elif std_mag < 0.5:
            scores["Stationary"] = 0.3
            scores["Walking"] = 0.6
            scores["Micro-movement"] = 0.4
        elif std_mag < 1.0:
            scores["Walking"] = 0.7
            scores["Shaking"] = 0.3
        else:
            scores["Running"] = 0.6
            scores["Shaking"] = 0.4
            scores["Vibration"] = 0.3
        
        # Peak-based events
        if peaks > 10:
            scores["Shaking"] = scores.get("Shaking", 0) + 0.3
            scores["Vibration"] = scores.get("Vibration", 0) + 0.2
        
        if peaks > 50:
            scores["Impact"] = 0.7
            scores["Sudden acceleration"] = 0.5
        
        # Frequency-based events
        if freq_info["dominant_freq"] > 5.0:
            scores["Periodic motion"] = 0.6
            scores["Vibration"] = scores.get("Vibration", 0) + 0.3
        
        # Magnitude-based events
        if mean_mag > 15.0:
            scores["Vehicle motion"] = 0.7
            scores["Sudden acceleration"] = 0.5
        
        if mean_mag < 8.0 and std_mag < 0.05:
            scores["Free fall"] = 0.6
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def classify_gyroscope_events(self, gyro_data: Dict[str, np.ndarray], 
                                 sample_rate: float) -> Dict[str, float]:
        """Classify gyroscope events."""
        if not gyro_data or 'x' not in gyro_data:
            return {"error": "No gyroscope data"}
        
        x = gyro_data['x']
        y = gyro_data['y']
        z = gyro_data['z']
        
        # Calculate magnitude
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        # Detect features
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        peaks = self.detect_peaks(magnitude)
        freq_info = self.detect_frequency_content(magnitude, sample_rate)
        
        scores = {}
        
        # Rotation detection
        if std_mag < 0.1:
            scores["Stationary"] = 0.9
            scores["Stable orientation"] = 0.7
        elif std_mag < 0.5:
            scores["Slow rotation"] = 0.7
            scores["Tilting"] = 0.4
        elif std_mag < 1.0:
            scores["Fast rotation"] = 0.6
            scores["Turning"] = 0.5
        else:
            scores["Spinning"] = 0.7
            scores["Fast rotation"] = 0.5
            scores["Shaking"] = 0.3
        
        # Peak-based events
        if peaks > 10:
            scores["Oscillation"] = 0.6
            scores["Twisting"] = 0.4
        
        if peaks > 50:
            scores["Intermittent rotation"] = 0.7
        
        # Frequency-based events
        if freq_info["dominant_freq"] > 3.0:
            scores["Continuous rotation"] = 0.6
            scores["Periodic motion"] = 0.4
        
        # Orientation change detection
        if mean_mag > 1.0 and std_mag > 0.5:
            scores["Orientation change"] = 0.7
            scores["Tilting"] = scores.get("Tilting", 0) + 0.3
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def classify_magnetometer_events(self, mag_data: Dict[str, np.ndarray], 
                                     sample_rate: float) -> Dict[str, float]:
        """Classify magnetometer events."""
        if not mag_data or 'x' not in mag_data:
            return {"error": "No magnetometer data"}
        
        x = mag_data['x']
        y = mag_data['y']
        z = mag_data['z']
        
        # Calculate magnitude
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        # Detect features
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        peaks = self.detect_peaks(magnitude)
        
        scores = {}
        
        # Field stability
        if std_mag < 1.0:
            scores["Stable field"] = 0.8
            scores["Normal variation"] = 0.6
        elif std_mag < 5.0:
            scores["Normal variation"] = 0.7
            scores["Field change"] = 0.4
        else:
            scores["Field change"] = 0.7
            scores["Magnetic interference"] = 0.6
        
        # Peak-based events
        if peaks > 5:
            scores["Orientation change"] = 0.6
            scores["Near metal"] = 0.4
        
        if peaks > 20:
            scores["Magnetic interference"] = scores.get("Magnetic interference", 0) + 0.3
            scores["Calibration needed"] = 0.5
        
        # Magnitude-based events
        if mean_mag > 60.0:
            scores["Field strength increase"] = 0.6
            scores["Near metal"] = scores.get("Near metal", 0) + 0.3
        
        if mean_mag < 20.0:
            scores["Field strength decrease"] = 0.6
        
        # Direction indicators (simplified)
        if len(x) > 0:
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            
            if mean_x > 0:
                scores["North-facing"] = 0.5
            else:
                scores["South-facing"] = 0.5
            
            if mean_y > 0:
                scores["East-facing"] = 0.5
            else:
                scores["West-facing"] = 0.5
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def classify_light_events(self, light_data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        """Classify light sensor events."""
        if light_data is None or len(light_data) == 0:
            return {"error": "No light data"}
        
        mean_light = np.mean(light_data)
        std_light = np.std(light_data)
        peaks = self.detect_peaks(light_data)
        
        scores = {}
        
        # Light level classification
        if mean_light < 10:
            scores["Dark"] = 0.9
        elif mean_light < 50:
            scores["Dim"] = 0.8
            scores["Dark"] = 0.4
        elif mean_light < 200:
            scores["Normal indoor"] = 0.8
            scores["Dim"] = 0.3
        elif mean_light < 500:
            scores["Bright indoor"] = 0.7
            scores["Normal indoor"] = 0.4
        elif mean_light < 1000:
            scores["Outdoor shade"] = 0.7
            scores["Bright indoor"] = 0.3
        elif mean_light < 10000:
            scores["Outdoor direct"] = 0.8
            scores["Outdoor shade"] = 0.3
        else:
            scores["Outdoor direct"] = 0.9
            scores["Flash"] = 0.4
        
        # Variation-based events
        if std_light > 100:
            scores["Variable"] = 0.7
            scores["Flicker"] = 0.4
        
        if peaks > 5:
            scores["Flash"] = scores.get("Flash", 0) + 0.4
            scores["Sudden increase"] = 0.6
        
        # Trend detection
        if len(light_data) > 10:
            trend = np.polyfit(range(len(light_data)), light_data, 1)[0]
            if trend > 1.0:
                scores["Gradual increase"] = 0.7
            elif trend < -1.0:
                scores["Gradual decrease"] = 0.7
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def classify_pressure_events(self, pressure_data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        """Classify pressure sensor events."""
        if pressure_data is None or len(pressure_data) == 0:
            return {"error": "No pressure data"}
        
        mean_pressure = np.mean(pressure_data)
        std_pressure = np.std(pressure_data)
        peaks = self.detect_peaks(pressure_data)
        
        scores = {}
        
        # Stability classification
        if std_pressure < 0.1:
            scores["Stable"] = 0.9
            scores["Normal variation"] = 0.6
        elif std_pressure < 0.5:
            scores["Normal variation"] = 0.7
            scores["Stable"] = 0.4
        else:
            scores["Weather pressure change"] = 0.6
            scores["Altitude change"] = 0.5
        
        # Peak-based events
        if peaks > 5:
            scores["Door opening"] = 0.5
            scores["Door closing"] = 0.5
        
        if peaks > 20:
            scores["Elevator use"] = 0.7
            scores["HVAC cycle"] = 0.4
        
        # Pressure level classification
        if mean_pressure < 80.0:
            scores["Extreme low"] = 0.7
            scores["Altitude change"] = scores.get("Altitude change", 0) + 0.3
        elif mean_pressure > 110.0:
            scores["Extreme high"] = 0.7
            scores["Pressure spike"] = 0.5
        
        # Trend detection
        if len(pressure_data) > 10:
            trend = np.polyfit(range(len(pressure_data)), pressure_data, 1)[0]
            if trend > 0.05:
                scores["Increasing"] = 0.7
            elif trend < -0.05:
                scores["Decreasing"] = 0.7
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def classify_location_events(self, location_data: Dict[str, np.ndarray], 
                                 sample_rate: float) -> Dict[str, float]:
        """Classify location events."""
        if not location_data or 'latitude' not in location_data:
            return {"error": "No location data"}
        
        lat = location_data['latitude']
        lon = location_data['longitude']
        altitude = location_data.get('altitude', np.array([]))
        speed = location_data.get('speed', np.array([]))
        
        scores = {}
        
        # Calculate movement
        if len(lat) > 1:
            lat_diff = np.diff(lat)
            lon_diff = np.diff(lon)
            movement = np.sqrt(lat_diff**2 + lon_diff**2)
            mean_movement = np.mean(movement)
            
            # Movement classification
            if mean_movement < 0.00001:
                scores["Stationary"] = 0.9
                scores["Indoor"] = 0.6
            elif mean_movement < 0.0001:
                scores["Walking"] = 0.7
                scores["Indoor"] = 0.4
            elif mean_movement < 0.001:
                scores["Running"] = 0.6
                scores["Cycling"] = 0.4
            else:
                scores["Driving"] = 0.7
                scores["Outdoor"] = 0.6
            
            # Movement variation
            movement_std = np.std(movement)
            if movement_std > 0.0005:
                scores["Route deviation"] = 0.6
                scores["Direction change"] = 0.5
        
        # Altitude analysis
        if len(altitude) > 1:
            alt_diff = np.diff(altitude)
            mean_alt_change = np.mean(np.abs(alt_diff))
            
            if mean_alt_change > 0.5:
                scores["Elevator motion"] = 0.7
                scores["Altitude change"] = 0.6
            elif mean_alt_change > 0.1:
                scores["Altitude change"] = 0.5
        
        # Speed analysis
        if len(speed) > 0:
            mean_speed = np.mean(speed)
            
            if mean_speed < 1.0:
                scores["Stationary"] = scores.get("Stationary", 0) + 0.3
            elif mean_speed < 3.0:
                scores["Walking"] = scores.get("Walking", 0) + 0.3
            elif mean_speed < 10.0:
                scores["Running"] = scores.get("Running", 0) + 0.3
                scores["Cycling"] = scores.get("Cycling", 0) + 0.3
            else:
                scores["Driving"] = scores.get("Driving", 0) + 0.3
        
        # GPS quality (based on movement consistency)
        if len(lat) > 10:
            movement_consistency = 1.0 / (np.std(movement) + 0.00001)
            if movement_consistency > 1000:
                scores["Stable GPS"] = 0.7
            elif movement_consistency < 100:
                scores["Drifting GPS"] = 0.6
                scores["GPS signal lost"] = 0.4
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def classify_all_sensor_events(self, sensor_data: Dict, sample_rates: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Classify events for all available sensors."""
        results = {}
        
        # Accelerometer events
        if 'accelerometer' in sensor_data:
            accel_sample_rate = sample_rates.get('accelerometer', 50.0)
            results['accelerometer'] = self.classify_accelerometer_events(
                sensor_data['accelerometer'], 
                accel_sample_rate
            )
        
        # Gyroscope events
        if 'gyroscope' in sensor_data:
            gyro_sample_rate = sample_rates.get('gyroscope', 50.0)
            results['gyroscope'] = self.classify_gyroscope_events(
                sensor_data['gyroscope'],
                gyro_sample_rate
            )
        
        # Magnetometer events
        if 'magnetometer' in sensor_data:
            mag_sample_rate = sample_rates.get('magnetometer', 50.0)
            results['magnetometer'] = self.classify_magnetometer_events(
                sensor_data['magnetometer'],
                mag_sample_rate
            )
        
        # Light events
        if 'light' in sensor_data:
            light_sample_rate = sample_rates.get('light', 10.0)
            results['light'] = self.classify_light_events(
                sensor_data['light']['samples'] if isinstance(sensor_data['light'], dict) else sensor_data['light'],
                light_sample_rate
            )
        
        # Pressure events
        if 'pressure' in sensor_data:
            pressure_sample_rate = sample_rates.get('pressure', 10.0)
            pressure_data = sensor_data['pressure']['samples'] if isinstance(sensor_data['pressure'], dict) else sensor_data['pressure']
            results['pressure'] = self.classify_pressure_events(
                pressure_data,
                pressure_sample_rate
            )
        
        # Location events
        if 'location' in sensor_data:
            location_sample_rate = sample_rates.get('location', 1.0)
            results['location'] = self.classify_location_events(
                sensor_data['location'],
                location_sample_rate
            )
        
        return results
