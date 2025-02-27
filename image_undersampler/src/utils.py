import os
import logging
import yaml
from datetime import datetime
from typing import Dict, Any
import torch

def setup_logging(log_dir: str = 'logs', level: str = 'DEBUG') -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('ImageUndersampler')
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        # Console handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        
        # File handler
        f_handler = logging.FileHandler(
            os.path.join(log_dir, f'undersampling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        )
        f_handler.setLevel(getattr(logging, level.upper()))
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device automatically if needed
    if config['model']['device'] == 'auto':
        config['model']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config

def check_system_requirements():
    """Check if system meets requirements"""
    requirements = {
        'gpu': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'torch_version': torch.__version__,
    }
    return requirements

def memory_stats() -> Dict[str, float]:
    """Get current memory usage statistics"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }