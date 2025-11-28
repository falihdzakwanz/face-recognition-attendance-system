"""
HEIC to JPEG Converter
======================

Module untuk convert file HEIC (Apple format) ke JPEG.
"""

from pathlib import Path
from typing import List, Tuple
from PIL import Image
import pillow_heif
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

# Register HEIF opener
pillow_heif.register_heif_opener()


def convert_heic_to_jpeg(
    source_dir: str,
    quality: int = 95,
    delete_original: bool = False
) -> Tuple[int, int, List[str]]:
    """
    Convert semua file HEIC di directory ke JPEG format.
    
    Args:
        source_dir: Directory yang berisi file HEIC
        quality: Quality JPEG output (1-100)
        delete_original: Hapus file HEIC original setelah convert
        
    Returns:
        Tuple of (total_files, converted_files, error_files)
        
    Example:
        >>> total, converted, errors = convert_heic_to_jpeg("dataset/Train")
        >>> print(f"Converted {converted}/{total} files")
    """
    logger = setup_logger("HEIC_Converter")
    
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return 0, 0, []
    
    logger.info(f"Scanning for HEIC files in: {source_dir}")
    
    # Find all HEIC files (case-insensitive)
    heic_extensions = ['.heic', '.HEIC', '.heif', '.HEIF']
    heic_files = []
    
    for ext in heic_extensions:
        heic_files.extend(source_path.rglob(f"*{ext}"))
    
    total_files = len(heic_files)
    
    if total_files == 0:
        logger.info("No HEIC files found.")
        return 0, 0, []
    
    logger.info(f"Found {total_files} HEIC files. Starting conversion...")
    
    converted = 0
    errors = []
    
    for heic_file in heic_files:
        try:
            # Open HEIC file
            img = Image.open(heic_file)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create output filename (replace extension with .jpg)
            output_file = heic_file.with_suffix('.jpg')
            
            # Save as JPEG
            img.save(output_file, 'JPEG', quality=quality, optimize=True)
            
            converted += 1
            logger.info(f"✓ Converted: {heic_file.name} -> {output_file.name}")
            
            # Delete original if requested
            if delete_original:
                heic_file.unlink()
                logger.info(f"  Deleted original: {heic_file.name}")
            
        except Exception as e:
            error_msg = f"Error converting {heic_file.name}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    logger.info("=" * 60)
    logger.info(f"Conversion Summary:")
    logger.info(f"  Total HEIC files: {total_files}")
    logger.info(f"  Successfully converted: {converted}")
    logger.info(f"  Errors: {len(errors)}")
    logger.info("=" * 60)
    
    if errors:
        logger.warning("Errors encountered:")
        for error in errors:
            logger.warning(f"  - {error}")
    
    return total_files, converted, errors


def scan_image_formats(directory: str) -> dict:
    """
    Scan directory untuk menghitung distribusi format image.
    
    Args:
        directory: Directory untuk scan
        
    Returns:
        Dictionary dengan format: {extension: count}
        
    Example:
        >>> formats = scan_image_formats("dataset/Train")
        >>> print(formats)
        {'.jpg': 250, '.png': 20, '.heic': 10}
    """
    logger = setup_logger("Format_Scanner")
    
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error(f"Directory not found: {directory}")
        return {}
    
    # Image extensions to look for
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', 
                       '.webp', '.heic', '.heif', '.JPG', '.JPEG', 
                       '.PNG', '.HEIC']
    
    format_count = {}
    
    for ext in image_extensions:
        files = list(dir_path.rglob(f"*{ext}"))
        if files:
            # Normalize extension to lowercase
            ext_lower = ext.lower()
            format_count[ext_lower] = format_count.get(ext_lower, 0) + len(files)
    
    logger.info(f"Image format distribution in {directory}:")
    for ext, count in sorted(format_count.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {ext}: {count} files")
    
    return format_count


if __name__ == "__main__":
    """
    Run HEIC converter pada dataset.
    
    Usage:
        python -m src.preprocessing.heic_converter
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert HEIC files to JPEG")
    parser.add_argument(
        "--source", 
        type=str, 
        default="dataset/Train",
        help="Source directory containing HEIC files"
    )
    parser.add_argument(
        "--quality", 
        type=int, 
        default=95,
        help="JPEG quality (1-100)"
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete original HEIC files after conversion"
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan formats without converting"
    )
    
    args = parser.parse_args()
    
    if args.scan_only:
        # Scan formats only
        formats = scan_image_formats(args.source)
        print(f"\nTotal formats found: {len(formats)}")
        
    else:
        # Convert HEIC files
        total, converted, errors = convert_heic_to_jpeg(
            args.source,
            quality=args.quality,
            delete_original=args.delete_original
        )
        
        if converted == total and total > 0:
            print("\n✓ All HEIC files converted successfully!")
        elif converted > 0:
            print(f"\n⚠ Converted {converted}/{total} files with {len(errors)} errors")
        else:
            print("\n✓ No HEIC files to convert")
