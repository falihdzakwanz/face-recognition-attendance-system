"""
Data Splitter
=============

Module untuk split dataset ke train/validation/test sets dengan stratified sampling.
"""

from pathlib import Path
from typing import Tuple, List
import shutil
import random
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.75,
    val_ratio: float = 0.0,
    test_ratio: float = 0.25,
    random_seed: int = 42,
    copy_files: bool = True
) -> Tuple[dict, dict, dict]:
    """
    Split dataset ke train/validation/test sets dengan stratified sampling.
    
    Args:
        source_dir: Source directory dengan subdirectories per student
        output_dir: Output root directory
        train_ratio: Proporsi data untuk training
        val_ratio: Proporsi data untuk validation
        test_ratio: Proporsi data untuk testing
        random_seed: Random seed untuk reproducibility
        copy_files: Copy files (True) atau move files (False)
        
    Returns:
        Tuple of (train_stats, val_stats, test_stats)
        
    Example:
        >>> train, val, test = split_dataset(
        ...     "dataset/Train_Aligned",
        ...     "dataset",
        ...     train_ratio=0.7,
        ...     val_ratio=0.15,
        ...     test_ratio=0.15
        ... )
    """
    logger = setup_logger("DataSplitter")
    
    # Validate ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    output_path = Path(output_dir)
    
    # Create output directories
    train_dir = output_path / "Train"
    val_dir = output_path / "Val"
    test_dir = output_path / "Test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    random.seed(random_seed)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    logger.info("=" * 60)
    logger.info("Dataset Splitting Configuration")
    logger.info("=" * 60)
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Split ratio: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Operation: {'Copy' if copy_files else 'Move'}")
    logger.info("=" * 60)
    
    train_stats = {'total_students': 0, 'total_images': 0, 'images_per_student': {}}
    val_stats = {'total_students': 0, 'total_images': 0, 'images_per_student': {}}
    test_stats = {'total_students': 0, 'total_images': 0, 'images_per_student': {}}
    
    # Get all student directories
    student_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if len(student_dirs) == 0:
        logger.error("No student directories found!")
        return train_stats, val_stats, test_stats
    
    logger.info(f"\nProcessing {len(student_dirs)} students...\n")
    
    # Process each student
    for student_dir in student_dirs:
        student_name = student_dir.name
        
        # Get all images for this student
        image_files = [
            f for f in student_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        num_images = len(image_files)
        
        if num_images == 0:
            logger.warning(f"⚠ {student_name}: No images found, skipping")
            continue
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate split indices using round for better distribution
        train_count = max(1, round(num_images * train_ratio))
        val_count = max(0, round(num_images * val_ratio))
        test_count = num_images - train_count - val_count
        
        # Adjust if needed (ensure at least 1 image in train, and all counts are non-negative)
        if test_count < 0:
            # If we over-allocated, reduce from validation first, then train
            if val_count > 0:
                val_count += test_count
                test_count = 0
            if val_count < 0:
                train_count += val_count
                val_count = 0
        
        # Split images
        train_images = image_files[:train_count]
        val_images = image_files[train_count:train_count + val_count]
        test_images = image_files[train_count + val_count:]
        
        # Create student directories in output
        train_student_dir = train_dir / student_name
        val_student_dir = val_dir / student_name
        test_student_dir = test_dir / student_name
        
        train_student_dir.mkdir(exist_ok=True)
        val_student_dir.mkdir(exist_ok=True)
        test_student_dir.mkdir(exist_ok=True)
        
        # Copy/move files
        operation = shutil.copy2 if copy_files else shutil.move
        
        for img in train_images:
            operation(str(img), str(train_student_dir / img.name))
        
        for img in val_images:
            operation(str(img), str(val_student_dir / img.name))
        
        for img in test_images:
            operation(str(img), str(test_student_dir / img.name))
        
        # Update statistics
        if train_count > 0:
            train_stats['total_students'] += 1
            train_stats['total_images'] += train_count
            train_stats['images_per_student'][student_name] = train_count
        
        if val_count > 0:
            val_stats['total_students'] += 1
            val_stats['total_images'] += val_count
            val_stats['images_per_student'][student_name] = val_count
        
        if test_count > 0:
            test_stats['total_students'] += 1
            test_stats['total_images'] += test_count
            test_stats['images_per_student'][student_name] = test_count
        
        logger.info(
            f"{student_name:40} | Total: {num_images:2} | "
            f"Train: {train_count:2} | Val: {val_count:2} | Test: {test_count:2}"
        )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Split Summary")
    logger.info("=" * 60)
    logger.info(f"TRAIN SET:")
    logger.info(f"  Students: {train_stats['total_students']}")
    logger.info(f"  Images: {train_stats['total_images']}")
    if train_stats['total_students'] > 0:
        logger.info(f"  Avg images/student: {train_stats['total_images']/train_stats['total_students']:.1f}")
    
    logger.info(f"\nVALIDATION SET:")
    logger.info(f"  Students: {val_stats['total_students']}")
    logger.info(f"  Images: {val_stats['total_images']}")
    if val_stats['total_students'] > 0:
        logger.info(f"  Avg images/student: {val_stats['total_images']/val_stats['total_students']:.1f}")
    
    logger.info(f"\nTEST SET:")
    logger.info(f"  Students: {test_stats['total_students']}")
    logger.info(f"  Images: {test_stats['total_images']}")
    if test_stats['total_students'] > 0:
        logger.info(f"  Avg images/student: {test_stats['total_images']/test_stats['total_students']:.1f}")
    logger.info("=" * 60)
    
    return train_stats, val_stats, test_stats


def verify_split(base_dir: str) -> dict:
    """
    Verify dataset split dan tampilkan statistik.
    
    Args:
        base_dir: Base directory yang berisi Train/Val/Test folders
        
    Returns:
        Dictionary dengan statistik per split
        
    Example:
        >>> stats = verify_split("dataset")
        >>> print(stats['train']['total_images'])
    """
    logger = setup_logger("DataVerifier")
    
    base_path = Path(base_dir)
    splits = ['Train', 'Val', 'Test']
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    all_stats = {}
    
    logger.info("Verifying dataset split...")
    
    for split in splits:
        split_dir = base_path / split
        
        if not split_dir.exists():
            logger.warning(f"{split} directory not found")
            continue
        
        student_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        total_images = 0
        images_per_student = {}
        
        for student_dir in student_dirs:
            images = [
                f for f in student_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ]
            num_images = len(images)
            total_images += num_images
            images_per_student[student_dir.name] = num_images
        
        all_stats[split.lower()] = {
            'total_students': len(student_dirs),
            'total_images': total_images,
            'avg_images_per_student': total_images / len(student_dirs) if student_dirs else 0,
            'images_per_student': images_per_student
        }
        
        logger.info(f"{split}: {len(student_dirs)} students, {total_images} images")
    
    return all_stats


if __name__ == "__main__":
    """
    Run dataset splitter.
    
    Usage:
        python -m src.preprocessing.data_splitter
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument(
        "--source",
        type=str,
        default="dataset/Train_Aligned",
        help="Source directory with aligned faces"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset",
        help="Output base directory"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Validation set ratio (0.0 to skip validation)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.25,
        help="Test set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing split"
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Verify existing split
        stats = verify_split(args.output)
        print("\n✓ Verification complete")
    else:
        # Perform split
        train, val, test = split_dataset(
            args.source,
            args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
            copy_files=not args.move
        )
        
        print("\n✓ Dataset split complete!")
