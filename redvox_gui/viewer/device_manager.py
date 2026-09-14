"""
GPU detection and automatic device selection for ML models.
Supports TensorFlow, PyTorch, and automatic CPU/GPU selection.
"""

import platform
import subprocess
from typing import Dict, List, Optional, Tuple


class DeviceManager:
    """Manage device detection and selection for ML models."""
    
    def __init__(self):
        self.available_devices = {}
        self.current_device = 'cpu'
        self._detect_devices()
    
    def _detect_devices(self):
        """Detect available computing devices."""
        self.available_devices = {
            'cpu': True,
            'cuda': False,
            'mps': False,  # Apple Silicon GPU
            'xla': False  # XLA for TPUs
        }
        
        # Detect CUDA (NVIDIA GPU)
        try:
            import torch
            if torch.cuda.is_available():
                self.available_devices['cuda'] = True
                self.available_devices['cuda_devices'] = torch.cuda.device_count()
                self.available_devices['cuda_device_names'] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
        except ImportError:
            pass
        
        # Detect MPS (Apple Silicon)
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.available_devices['mps'] = True
        except ImportError:
            pass
        
        # Detect TensorFlow GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.available_devices['cuda'] = True
                self.available_devices['tf_gpus'] = len(gpus)
        except ImportError:
            pass
    
    def get_available_devices(self) -> Dict:
        """Get dictionary of available devices."""
        return self.available_devices
    
    def get_optimal_device(self, prefer_gpu: bool = True) -> str:
        """
        Get optimal device for computation.
        
        Args:
            prefer_gpu: Whether to prefer GPU over CPU
        
        Returns:
            Device string ('cuda', 'mps', 'cpu')
        """
        if prefer_gpu:
            if self.available_devices.get('cuda', False):
                return 'cuda'
            elif self.available_devices.get('mps', False):
                return 'mps'
        
        return 'cpu'
    
    def set_device(self, device: str):
        """
        Set current device.
        
        Args:
            device: Device string ('cuda', 'mps', 'cpu')
        """
        if device in self.available_devices and self.available_devices[device]:
            self.current_device = device
        else:
            print(f"Device {device} not available, using CPU")
            self.current_device = 'cpu'
    
    def get_current_device(self) -> str:
        """Get current device."""
        return self.current_device
    
    def get_device_info(self) -> Dict:
        """Get detailed device information."""
        info = {
            'current_device': self.current_device,
            'available_devices': self.available_devices,
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor()
        }
        
        # Add GPU details if available
        if self.available_devices.get('cuda', False):
            try:
                import torch
                info['cuda_version'] = torch.version.cuda
                info['torch_version'] = torch.__version__
            except ImportError:
                pass
            
            try:
                import tensorflow as tf
                info['tf_version'] = tf.__version__
            except ImportError:
                pass
        
        return info
    
    def auto_select_device(self, task_type: str = 'inference') -> str:
        """
        Automatically select device based on task type.
        
        Args:
            task_type: Type of task ('inference', 'training', 'large_model')
        
        Returns:
            Selected device string
        """
        # For inference, prefer GPU but CPU is acceptable
        if task_type == 'inference':
            if self.available_devices.get('cuda', False):
                return 'cuda'
            elif self.available_devices.get('mps', False):
                return 'mps'
            return 'cpu'
        
        # For training, strongly prefer GPU
        elif task_type == 'training':
            if self.available_devices.get('cuda', False):
                return 'cuda'
            elif self.available_devices.get('mps', False):
                return 'mps'
            print("Warning: Training on CPU will be slow")
            return 'cpu'
        
        # For large models, prefer GPU with more memory
        elif task_type == 'large_model':
            if self.available_devices.get('cuda', False):
                return 'cuda'
            elif self.available_devices.get('mps', False):
                return 'mps'
            print("Warning: Large model on CPU may be very slow")
            return 'cpu'
        
        return 'cpu'
    
    def get_memory_info(self) -> Dict:
        """Get memory information for available devices."""
        memory_info = {}
        
        # GPU memory
        if self.available_devices.get('cuda', False):
            try:
                import torch
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_info[f'cuda_{i}'] = {
                        'total_memory_gb': props.total_memory / (1024**3),
                        'name': props.name
                    }
            except ImportError:
                pass
        
        # System memory
        try:
            import psutil
            memory_info['system'] = {
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3)
            }
        except ImportError:
            pass
        
        return memory_info


class ModelDeviceWrapper:
    """Wrapper for running models on specific devices."""
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.current_device = device_manager.get_current_device()
    
    def to_device(self, model, device: Optional[str] = None):
        """
        Move model to specified device.
        
        Args:
            model: PyTorch or TensorFlow model
            device: Device string (uses current if None)
        
        Returns:
            Model on specified device
        """
        target_device = device or self.current_device
        
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                if target_device == 'cuda':
                    return model.cuda()
                elif target_device == 'mps':
                    return model.to('mps')
                else:
                    return model.cpu()
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            if target_device == 'cuda':
                with tf.device('/GPU:0'):
                    return model
            else:
                with tf.device('/CPU:0'):
                    return model
        except ImportError:
            pass
        
        return model
    
    def get_model_device(self, model) -> str:
        """
        Get the device a model is currently on.
        
        Args:
            model: PyTorch model
        
        Returns:
            Device string
        """
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                device = next(model.parameters()).device
                if device.type == 'cuda':
                    return 'cuda'
                elif device.type == 'mps':
                    return 'mps'
                else:
                    return 'cpu'
        except ImportError:
            pass
        
        return 'cpu'


def get_device_manager() -> DeviceManager:
    """Get singleton device manager instance."""
    if not hasattr(get_device_manager, '_instance'):
        get_device_manager._instance = DeviceManager()
    return get_device_manager._instance


def auto_select_device_for_task(task_type: str = 'inference') -> str:
    """
    Convenience function to auto-select device for a task.
    
    Args:
        task_type: Type of task ('inference', 'training', 'large_model')
    
    Returns:
        Selected device string
    """
    manager = get_device_manager()
    return manager.auto_select_device(task_type)
