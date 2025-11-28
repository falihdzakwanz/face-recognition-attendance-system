"""
Setup Verification Script
=========================

Script untuk verify bahwa semua dependencies dan modules terinstall dengan benar.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def test_imports():
    """Test semua import yang diperlukan."""
    print("Testing imports...")
    
    tests = []
    
    # Core libraries
    try:
        import yaml
        tests.append(("✓", "yaml (PyYAML)"))
    except ImportError:
        tests.append(("✗", "yaml (PyYAML) - pip install pyyaml"))
    
    try:
        import cv2
        tests.append(("✓", f"opencv-python {cv2.__version__}"))
    except ImportError:
        tests.append(("✗", "opencv-python - pip install opencv-python"))
    
    try:
        import numpy as np
        tests.append(("✓", f"numpy {np.__version__}"))
    except ImportError:
        tests.append(("✗", "numpy - pip install numpy"))
    
    try:
        import PIL
        tests.append(("✓", f"Pillow {PIL.__version__}"))
    except ImportError:
        tests.append(("✗", "Pillow - pip install Pillow"))
    
    try:
        import pillow_heif
        tests.append(("✓", "pillow-heif"))
    except ImportError:
        tests.append(("✗", "pillow-heif - pip install pillow-heif"))
    
    try:
        from mtcnn import MTCNN
        tests.append(("✓", "mtcnn"))
    except ImportError:
        tests.append(("✗", "mtcnn - pip install mtcnn tensorflow"))
    
    try:
        import albumentations
        tests.append(("✓", f"albumentations {albumentations.__version__}"))
    except ImportError:
        tests.append(("✗", "albumentations - pip install albumentations"))
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "CUDA" if cuda_available else "CPU"
        tests.append(("✓", f"PyTorch {torch.__version__} ({device})"))
    except ImportError:
        tests.append(("✗", "PyTorch - pip install torch torchvision"))
    
    try:
        import tensorflow as tf
        tests.append(("✓", f"TensorFlow {tf.__version__}"))
    except ImportError:
        tests.append(("✗", "TensorFlow - pip install tensorflow"))
    
    try:
        import timm
        tests.append(("✓", f"timm {timm.__version__}"))
    except ImportError:
        tests.append(("✗", "timm - pip install timm"))
    
    try:
        import matplotlib
        tests.append(("✓", f"matplotlib {matplotlib.__version__}"))
    except ImportError:
        tests.append(("✗", "matplotlib - pip install matplotlib"))
    
    try:
        import seaborn
        tests.append(("✓", f"seaborn {seaborn.__version__}"))
    except ImportError:
        tests.append(("✗", "seaborn - pip install seaborn"))
    
    try:
        import pandas as pd
        tests.append(("✓", f"pandas {pd.__version__}"))
    except ImportError:
        tests.append(("✗", "pandas - pip install pandas"))
    
    try:
        import sklearn
        tests.append(("✓", f"scikit-learn {sklearn.__version__}"))
    except ImportError:
        tests.append(("✗", "scikit-learn - pip install scikit-learn"))
    
    # Print results
    print("\n" + "=" * 60)
    print("DEPENDENCY CHECK")
    print("=" * 60)
    
    for status, lib in tests:
        print(f"{status} {lib}")
    
    failed = [lib for status, lib in tests if status == "✗"]
    
    print("=" * 60)
    
    if failed:
        print(f"\n⚠ {len(failed)} dependencies missing!")
        print("\nTo install all dependencies:")
        print("pip install -r requirements.txt")
        return False
    else:
        print(f"\n✓ All {len(tests)} dependencies installed!")
        return True


def test_modules():
    """Test custom modules."""
    print("\n" + "=" * 60)
    print("MODULE CHECK")
    print("=" * 60)
    
    tests = []
    
    try:
        from utils.config_loader import load_config
        tests.append(("✓", "utils.config_loader"))
    except Exception as e:
        tests.append(("✗", f"utils.config_loader - {e}"))
    
    try:
        from utils.logger import setup_logger
        tests.append(("✓", "utils.logger"))
    except Exception as e:
        tests.append(("✗", f"utils.logger - {e}"))
    
    try:
        from utils.visualization import plot_training_history
        tests.append(("✓", "utils.visualization"))
    except Exception as e:
        tests.append(("✗", f"utils.visualization - {e}"))
    
    try:
        from preprocessing.heic_converter import convert_heic_to_jpeg
        tests.append(("✓", "preprocessing.heic_converter"))
    except Exception as e:
        tests.append(("✗", f"preprocessing.heic_converter - {e}"))
    
    try:
        from preprocessing.face_detector import FaceDetector
        tests.append(("✓", "preprocessing.face_detector"))
    except Exception as e:
        tests.append(("✗", f"preprocessing.face_detector - {e}"))
    
    try:
        from preprocessing.data_splitter import split_dataset
        tests.append(("✓", "preprocessing.data_splitter"))
    except Exception as e:
        tests.append(("✗", f"preprocessing.data_splitter - {e}"))
    
    try:
        from preprocessing.augmentation import create_augmentation_pipeline
        tests.append(("✓", "preprocessing.augmentation"))
    except Exception as e:
        tests.append(("✗", f"preprocessing.augmentation - {e}"))
    
    # Print results
    for status, module in tests:
        print(f"{status} {module}")
    
    failed = [mod for status, mod in tests if status == "✗"]
    
    print("=" * 60)
    
    if failed:
        print(f"\n⚠ {len(failed)} modules have issues!")
        return False
    else:
        print(f"\n✓ All {len(tests)} modules working!")
        return True


def test_config():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("CONFIGURATION CHECK")
    print("=" * 60)
    
    try:
        from utils.config_loader import load_config, get_paths
        
        config = load_config("config.yaml")
        print("✓ config.yaml loaded successfully")
        
        print(f"  Project: {config['project']['name']}")
        print(f"  Version: {config['project']['version']}")
        
        paths = get_paths(config)
        print(f"✓ Paths extracted: {len(paths)} paths")
        
        print("=" * 60)
        print("\n✓ Configuration OK!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        print("=" * 60)
        return False


def test_dataset():
    """Test dataset structure."""
    print("\n" + "=" * 60)
    print("DATASET CHECK")
    print("=" * 60)
    
    dataset_path = Path("dataset/Train")
    
    if not dataset_path.exists():
        print(f"✗ Dataset directory not found: {dataset_path}")
        print("=" * 60)
        return False
    
    print(f"✓ Dataset directory found: {dataset_path}")
    
    # Count students
    student_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"✓ Found {len(student_dirs)} student directories")
    
    # Count images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic'}
    total_images = 0
    
    for student_dir in student_dirs:
        images = [f for f in student_dir.iterdir() if f.suffix.lower() in image_extensions]
        total_images += len(images)
    
    print(f"✓ Found {total_images} total images")
    
    if total_images > 0:
        avg_images = total_images / len(student_dirs)
        print(f"✓ Average {avg_images:.1f} images per student")
    
    print("=" * 60)
    print("\n✓ Dataset structure OK!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("FACE RECOGNITION SYSTEM - SETUP VERIFICATION")
    print("=" * 60)
    
    results = {
        "Dependencies": test_imports(),
        "Modules": test_modules(),
        "Configuration": test_config(),
        "Dataset": test_dataset()
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print("=" * 60)
    
    if all(results.values()):
        print("\n🎉 All checks passed! System ready to use.")
        print("\nNext steps:")
        print("  1. python main.py --preprocess")
        print("  2. Implement model architectures")
        print("  3. python main.py --train")
        return 0
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
