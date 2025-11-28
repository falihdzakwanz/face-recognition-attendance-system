"""
Configuration Loader
====================

Module untuk load dan parse configuration dari YAML file.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration dari YAML file.
    
    Args:
        config_path: Path ke config file (default: config.yaml)
    
    Returns:
        Dictionary berisi konfigurasi
        
    Raises:
        FileNotFoundError: Jika config file tidak ditemukan
        yaml.YAMLError: Jika terjadi error parsing YAML
        
    Example:
        >>> config = load_config()
        >>> print(config['project']['name'])
        Face Recognition Presensi Mahasiswa
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate essential keys
        required_keys = ['project', 'paths', 'preprocessing', 'cnn_model', 
                        'transformer_model', 'evaluation', 'system']
        
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required keys in config: {missing_keys}")
        
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config: {e}")


def get_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Extract dan convert semua paths ke Path objects.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary dengan Path objects
        
    Example:
        >>> config = load_config()
        >>> paths = get_paths(config)
        >>> print(paths['dataset_root'])
        dataset
    """
    paths_config = config.get('paths', {})
    
    return {
        key: Path(value) 
        for key, value in paths_config.items()
    }


def create_directories(paths: Dict[str, Path]) -> None:
    """
    Create directories yang diperlukan jika belum ada.
    
    Args:
        paths: Dictionary berisi Path objects
        
    Example:
        >>> config = load_config()
        >>> paths = get_paths(config)
        >>> create_directories(paths)
    """
    # Directories yang perlu dibuat (exclude input directories)
    output_dirs = ['output_dir', 'models_dir', 'logs_dir', 
                   'val_dir', 'test_dir']
    
    for key in output_dirs:
        if key in paths:
            path = paths[key]
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {path}")


if __name__ == "__main__":
    # Test configuration loader
    try:
        config = load_config()
        print("✓ Configuration loaded successfully!")
        print(f"  Project: {config['project']['name']}")
        print(f"  Version: {config['project']['version']}")
        
        paths = get_paths(config)
        print(f"\n✓ Paths extracted: {len(paths)} paths")
        
        create_directories(paths)
        print("\n✓ All directories created!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
