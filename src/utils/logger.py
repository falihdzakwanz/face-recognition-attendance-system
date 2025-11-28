"""
Logger Utilities
================

Module untuk setup logging sistem dengan format yang konsisten.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "FaceRecognition",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger dengan konfigurasi standar.
    
    Args:
        name: Nama logger
        log_level: Level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path ke log file (optional). Jika None, hanya log ke console
        log_format: Custom format string (optional)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger("Training", log_level="DEBUG")
        >>> logger.info("Training started")
        2025-11-25 10:30:00 - Training - INFO - Training started
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_experiment_name(prefix: str = "exp") -> str:
    """
    Generate experiment name dengan timestamp.
    
    Args:
        prefix: Prefix untuk experiment name
        
    Returns:
        Experiment name dengan format: prefix_YYYYMMDD_HHMMSS
        
    Example:
        >>> exp_name = get_experiment_name("cnn_training")
        >>> print(exp_name)
        cnn_training_20251125_103045
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def log_config(logger: logging.Logger, config: dict, title: str = "Configuration") -> None:
    """
    Log configuration dictionary dengan format yang rapi.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        title: Title untuk log section
        
    Example:
        >>> logger = setup_logger()
        >>> config = {'batch_size': 32, 'lr': 0.001}
        >>> log_config(logger, config, "Training Config")
    """
    logger.info("=" * 60)
    logger.info(f"{title:^60}")
    logger.info("=" * 60)
    
    def _log_dict(d: dict, indent: int = 0):
        """Helper function untuk log nested dictionary"""
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                _log_dict(value, indent + 1)
            else:
                logger.info(f"{prefix}{key}: {value}")
    
    _log_dict(config)
    logger.info("=" * 60)


class LoggerContext:
    """
    Context manager untuk temporary logging ke file.
    
    Example:
        >>> logger = setup_logger()
        >>> with LoggerContext(logger, "training.log"):
        ...     logger.info("This will be logged to file")
    """
    
    def __init__(self, logger: logging.Logger, log_file: str):
        """
        Initialize context manager.
        
        Args:
            logger: Logger instance
            log_file: Path ke log file
        """
        self.logger = logger
        self.log_file = log_file
        self.file_handler = None
    
    def __enter__(self):
        """Setup file handler"""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        self.file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove file handler"""
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()


if __name__ == "__main__":
    # Test logger
    logger = setup_logger("Test", log_level="DEBUG")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test experiment name
    exp_name = get_experiment_name("test")
    print(f"\nExperiment name: {exp_name}")
    
    # Test config logging
    test_config = {
        'model': {'name': 'FaceNet', 'layers': 50},
        'training': {'batch_size': 32, 'epochs': 50}
    }
    log_config(logger, test_config, "Test Configuration")
