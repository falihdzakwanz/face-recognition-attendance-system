"""
Data Augmentation
=================

Module untuk data augmentation menggunakan Albumentations library.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger


def create_augmentation_pipeline(
    mode: str = "train",
    target_size: int = 160,
    normalize: bool = True
) -> A.Compose:
    """
    Create augmentation pipeline untuk training atau testing.
    
    Args:
        mode: "train" atau "test"
        target_size: Target image size
        normalize: Normalize pixel values ke [0, 1] atau [-1, 1]
        
    Returns:
        Albumentations Compose pipeline
        
    Example:
        >>> train_aug = create_augmentation_pipeline(mode="train")
        >>> augmented = train_aug(image=image)['image']
    """
    if mode == "train":
        # Training augmentation - heavy augmentation untuk dataset kecil
        transforms = [
            # Resize
            A.Resize(target_size, target_size),
            
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                p=0.6
            ),
            
            # Perspective & distortion
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),
            
            # Color transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.4
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.4
            ),
            
            # Convert to grayscale occasionally (simulate poor lighting)
            A.ToGray(p=0.15),
            
            # Channel operations
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
            A.ChannelShuffle(p=0.1),
            
            # Blur and noise
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.4),
            
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.4),
            
            # Lighting & shadows
            A.RandomGamma(gamma_limit=(70, 130), p=0.4),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                p=0.2
            ),
            A.RandomToneCurve(scale=0.1, p=0.3),
            
            # Compression artifacts (simulate low quality images)
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            
            # Cutout (simulates occlusion)
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(int(target_size * 0.05), int(target_size * 0.15)),
                hole_width_range=(int(target_size * 0.05), int(target_size * 0.15)),
                fill_value=128,
                p=0.3
            ),
            
            # CLAHE for better contrast
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # Sharpening
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
        ]
        
    elif mode == "test" or mode == "val":
        # Test/validation - minimal augmentation
        transforms = [
            A.Resize(target_size, target_size),
        ]
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'test', or 'val'")
    
    # Add normalization
    if normalize:
        transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],   # ImageNet std
                max_pixel_value=255.0
            )
        )
    
    # Add ToTensor (for PyTorch)
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def augment_and_save(
    image_path: str,
    output_dir: str,
    num_augmentations: int = 5,
    augmentation_pipeline: Optional[A.Compose] = None
) -> list:
    """
    Generate augmented versions dari single image dan save.
    
    Args:
        image_path: Path ke input image
        output_dir: Output directory
        num_augmentations: Number of augmented versions to create
        augmentation_pipeline: Custom pipeline (optional)
        
    Returns:
        List of paths ke augmented images
        
    Example:
        >>> paths = augment_and_save(
        ...     "dataset/Train/Student/photo.jpg",
        ...     "dataset/Train_Augmented/Student",
        ...     num_augmentations=5
        ... )
    """
    logger = setup_logger("Augmentation")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return []
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create augmentation pipeline if not provided
    if augmentation_pipeline is None:
        augmentation_pipeline = create_augmentation_pipeline(mode="train", normalize=False)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get original filename
    original_name = Path(image_path).stem
    original_ext = Path(image_path).suffix
    
    saved_paths = []
    
    # Generate augmented images
    for i in range(num_augmentations):
        try:
            # Apply augmentation
            augmented = augmentation_pipeline(image=image)['image']
            
            # Convert back to numpy if it's a tensor
            if hasattr(augmented, 'numpy'):
                augmented = augmented.numpy().transpose(1, 2, 0)
                augmented = (augmented * 255).astype(np.uint8)
            
            # Save augmented image
            output_file = output_path / f"{original_name}_aug{i+1}{original_ext}"
            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), augmented_bgr)
            
            saved_paths.append(str(output_file))
            
        except Exception as e:
            logger.error(f"Error augmenting {image_path}: {str(e)}")
    
    return saved_paths


def visualize_augmentations(
    image_path: str,
    num_samples: int = 6,
    save_path: Optional[str] = None
):
    """
    Visualize augmentations untuk debugging.
    
    Args:
        image_path: Path ke input image
        num_samples: Number of augmented samples to show
        save_path: Path untuk save visualization (optional)
    """
    import matplotlib.pyplot as plt
    
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create augmentation pipeline
    aug_pipeline = create_augmentation_pipeline(mode="train", normalize=False)
    
    # Create figure
    rows = (num_samples + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, rows * 4))
    axes = axes.flatten()
    
    # Show original
    axes[0].imshow(image)
    axes[0].set_title("Original", fontweight='bold')
    axes[0].axis('off')
    
    # Show augmented versions
    for i in range(1, num_samples):
        augmented = aug_pipeline(image=image)['image']
        
        # Convert tensor to numpy if needed
        if hasattr(augmented, 'numpy'):
            augmented = augmented.numpy().transpose(1, 2, 0)
            augmented = (augmented * 255).astype(np.uint8)
        
        axes[i].imshow(augmented)
        axes[i].set_title(f"Augmented {i}", fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


class AugmentationConfig:
    """
    Configuration class untuk augmentation parameters.
    """
    
    # CNN config (FaceNet - 160x160)
    CNN_CONFIG = {
        'target_size': 160,
        'normalize': True,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    
    # Transformer config (ViT/DeiT - 224x224)
    TRANSFORMER_CONFIG = {
        'target_size': 224,
        'normalize': True,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }


if __name__ == "__main__":
    """
    Test augmentation pipeline dan visualize results.
    
    Usage:
        python -m src.preprocessing.augmentation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data augmentation")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to test image"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
        help="Number of augmented samples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save visualization"
    )
    
    args = parser.parse_args()
    
    print(f"Visualizing augmentations for: {args.image}")
    print(f"Generating {args.samples} augmented samples...")
    
    visualize_augmentations(
        args.image,
        num_samples=args.samples,
        save_path=args.output
    )
    
    print("\n✓ Augmentation test complete!")
