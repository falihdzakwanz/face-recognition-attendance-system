"""
Main Entry Point
================

Main script untuk menjalankan face recognition pipeline:
1. Preprocessing (HEIC conversion, face detection, data splitting)
2. Model training (CNN & Transformer)
3. Evaluation & comparison
4. Desktop application

Usage:
    python main.py --help
"""

import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import load_config, get_paths, create_directories
from utils.logger import setup_logger
from preprocessing.heic_converter import convert_heic_to_jpeg, scan_image_formats
from preprocessing.face_detector import detect_and_align_faces
from preprocessing.data_splitter import split_dataset, verify_split


def run_preprocessing(config: dict, args: argparse.Namespace):
    """
    Run data preprocessing pipeline.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    logger = setup_logger("Preprocessing")
    
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("=" * 60)
    
    paths = get_paths(config)
    train_dir = str(paths['train_dir'])
    
    # Step 1.1: Scan image formats
    if not args.skip_scan:
        logger.info("\n1.1 Scanning image formats...")
        formats = scan_image_formats(train_dir)
        
        if not formats:
            logger.error("No images found in dataset!")
            return False
    
    # Step 1.2: Convert HEIC files
    if not args.skip_heic:
        logger.info("\n1.2 Converting HEIC files to JPEG...")
        total, converted, errors = convert_heic_to_jpeg(
            train_dir,
            quality=95,
            delete_original=args.delete_heic
        )
        
        if errors:
            logger.warning(f"Some HEIC conversions failed: {len(errors)} errors")
    
    # Step 1.3: Face detection and alignment
    if not args.skip_detection:
        logger.info("\n1.3 Detecting and aligning faces...")
        
        aligned_dir = str(Path(train_dir).parent / "Train_Aligned")
        
        total, success, failed = detect_and_align_faces(
            train_dir,
            aligned_dir,
            target_size=(
                config['preprocessing']['face_alignment']['target_size'][0],
                config['preprocessing']['face_alignment']['target_size'][1]
            ),
            min_confidence=config['preprocessing']['face_detection']['confidence_threshold']
        )
        
        if success == 0:
            logger.error("No faces detected! Check your dataset.")
            return False
        
        if failed > total * 0.1:  # More than 10% failed
            logger.warning(f"High failure rate: {failed}/{total} images failed")
    
    # Step 1.4: Split dataset
    if not args.skip_split:
        logger.info("\n1.4 Splitting dataset into train/val/test...")
        
        aligned_dir = str(Path(train_dir).parent / "Train_Aligned")
        output_dir = str(paths['dataset_root'])
        
        train_stats, val_stats, test_stats = split_dataset(
            aligned_dir,
            output_dir,
            train_ratio=config['preprocessing']['data_split']['train_ratio'],
            val_ratio=config['preprocessing']['data_split']['val_ratio'],
            test_ratio=config['preprocessing']['data_split']['test_ratio'],
            random_seed=config['preprocessing']['data_split']['random_seed'],
            copy_files=True
        )
        
        # Verify split
        logger.info("\n1.5 Verifying dataset split...")
        verify_split(output_dir)
    
    logger.info("\n✓ Preprocessing completed successfully!")
    return True


def run_training(config: dict, args: argparse.Namespace):
    """
    Run model training.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    logger = setup_logger("Training")
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("=" * 60)
    
    # Import training modules
    try:
        from training.train_cnn import train_cnn_model
        from training.train_transformer import train_transformer_model
    except ImportError as e:
        logger.error(f"Training modules not yet implemented: {e}")
        logger.info("Please implement training modules first.")
        return False
    
    # Train CNN model
    if not args.skip_cnn:
        logger.info("\n2.1 Training CNN Model (FaceNet)...")
        
        # Check if CNN model already trained
        models_dir = Path(config['paths']['models_dir'])
        cnn_models = list(models_dir.glob("cnn_arcface_*/best_model.pth"))
        
        if cnn_models and args.use_existing:
            latest_cnn = max(cnn_models, key=lambda p: p.parent.name)
            logger.info(f"Found existing CNN model: {latest_cnn.parent.name}")
            logger.info("Skipping CNN training (use --no-use-existing to retrain)")
        else:
            try:
                cnn_model = train_cnn_model(config)
                logger.info("✓ CNN training completed!")
            except Exception as e:
                logger.error(f"CNN training failed: {e}")
                if not args.continue_on_error:
                    return False
    
    # Train Transformer model
    if not args.skip_transformer:
        logger.info("\n2.2 Training Transformer Model (DeiT)...")
        
        # Check if Transformer model already trained
        models_dir = Path(config['paths']['models_dir'])
        transformer_models = list(models_dir.glob("transformer_deit_*/best_model.pth"))
        
        if transformer_models and args.use_existing:
            latest_transformer = max(transformer_models, key=lambda p: p.parent.name)
            logger.info(f"Found existing Transformer model: {latest_transformer.parent.name}")
            logger.info("Skipping Transformer training (use --no-use-existing to retrain)")
        else:
            try:
                transformer_model = train_transformer_model(config)
                logger.info("✓ Transformer training completed!")
            except Exception as e:
                logger.error(f"Transformer training failed: {e}")
                if not args.continue_on_error:
                    return False
    
    logger.info("\n✓ Training completed successfully!")
    return True


def run_evaluation(config: dict, args: argparse.Namespace):
    """
    Run model evaluation and comparison.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    logger = setup_logger("Evaluation")
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Import evaluation module
    try:
        from evaluation.evaluate import evaluate_models
    except ImportError as e:
        logger.error(f"Evaluation module not yet implemented: {e}")
        logger.info("Please implement evaluation module first.")
        return False
    
    try:
        results = evaluate_models(config)
        logger.info("✓ Evaluation completed!")
        return True
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False


def run_application(config: dict, args: argparse.Namespace):
    """
    Launch desktop application.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    logger = setup_logger("Application")
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: LAUNCHING DESKTOP APPLICATION")
    logger.info("=" * 60)
    
    framework = config['desktop_app']['framework']
    
    try:
        if framework == "gradio":
            from src.app.gradio_app import launch_gradio_app
            launch_gradio_app(config)
        elif framework == "pyqt5":
            from src.app.pyqt_app import launch_pyqt_app
            launch_pyqt_app(config)
        else:
            logger.error(f"Unknown framework: {framework}")
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"Application module not yet implemented: {e}")
        logger.info("Please implement application module first.")
        return False
    except Exception as e:
        logger.error(f"Application launch failed: {e}")
        return False


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(
        description="Face Recognition System untuk Presensi Mahasiswa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --all
  
  # Run only preprocessing
  python main.py --preprocess
  
  # Run training (skip preprocessing)
  python main.py --train --skip-preprocess
  
  # Launch application
  python main.py --app
  
  # Custom config file
  python main.py --config custom_config.yaml --all
        """
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    # Pipeline stages
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing only")
    parser.add_argument("--train", action="store_true", help="Run training only")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    parser.add_argument("--app", action="store_true", help="Launch application only")
    
    # Skip options for preprocessing
    parser.add_argument("--skip-scan", action="store_true", help="Skip image format scanning")
    parser.add_argument("--skip-heic", action="store_true", help="Skip HEIC conversion")
    parser.add_argument("--skip-detection", action="store_true", help="Skip face detection")
    parser.add_argument("--skip-split", action="store_true", help="Skip dataset splitting")
    parser.add_argument("--delete-heic", action="store_true", help="Delete HEIC files after conversion")
    
    # Skip options for training
    parser.add_argument("--skip-cnn", action="store_true", help="Skip CNN training")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip Transformer training")
    parser.add_argument("--use-existing", action="store_true", default=True, 
                        help="Use existing trained models if available (default: True)")
    parser.add_argument("--no-use-existing", action="store_false", dest="use_existing",
                        help="Force retrain even if models exist")
    
    # Other options
    parser.add_argument("--continue-on-error", action="store_true", help="Continue on error")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger = setup_logger("Main")
        
        logger.info("=" * 60)
        logger.info(f"{config['project']['name']:^60}")
        logger.info(f"Version {config['project']['version']:^60}")
        logger.info("=" * 60)
        
        # Create necessary directories
        paths = get_paths(config)
        create_directories(paths)
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Determine what to run
    run_all = args.all
    run_preprocess = args.preprocess or run_all
    run_train = args.train or run_all
    run_eval = args.evaluate or run_all
    run_app = args.app
    
    # If nothing specified, show help
    if not (run_preprocess or run_train or run_eval or run_app):
        parser.print_help()
        return 0
    
    # Run pipeline stages
    success = True
    
    if run_preprocess:
        success = run_preprocessing(config, args)
        if not success and not args.continue_on_error:
            logger.error("Preprocessing failed. Stopping pipeline.")
            return 1
    
    if run_train:
        success = run_training(config, args)
        if not success and not args.continue_on_error:
            logger.error("Training failed. Stopping pipeline.")
            return 1
    
    if run_eval:
        success = run_evaluation(config, args)
        if not success and not args.continue_on_error:
            logger.error("Evaluation failed. Stopping pipeline.")
            return 1
    
    if run_app:
        success = run_application(config, args)
        if not success:
            logger.error("Application launch failed.")
            return 1
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
