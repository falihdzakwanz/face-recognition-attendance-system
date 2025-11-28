"""
PyTorch Dataset Loader
======================

Custom dataset loader untuk face recognition dengan support untuk
data augmentation dan multi-output (embeddings & classification).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger


class FaceRecognitionDataset(Dataset):
    """
    Dataset untuk face recognition.

    Attributes:
        root_dir: Root directory berisi subdirectories per student
        transform: Augmentation pipeline
        class_to_idx: Mapping dari class name ke index
        samples: List of (image_path, class_idx) tuples

    Example:
        >>> dataset = FaceRecognitionDataset(
        ...     root_dir="dataset/Train",
        ...     transform=train_transform
        ... )
        >>> image, label = dataset[0]
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[A.Compose] = None,
        return_path: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory dengan subdirectories per student
            transform: Albumentations transform pipeline
            return_path: Return image path along with image and label
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_path = return_path
        self.logger = setup_logger("Dataset")

        # Image extensions
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Build dataset
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.logger.info(
            f"Dataset loaded: {len(self.samples)} images, {len(self.classes)} classes"
        )

    def _find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """Find classes (student names) dari directory structure."""
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self) -> List[Tuple[Path, int]]:
        """Build list of (image_path, class_idx) tuples."""
        samples = []

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.image_extensions:
                    samples.append((img_path, class_idx))

        if len(samples) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}")

        return samples

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Tuple of (image_tensor, class_idx) atau (image_tensor, class_idx, path)
        """
        img_path, class_idx = self.samples[idx]

        # Read image with error handling
        image = cv2.imread(str(img_path))
        
        # Check if image is valid
        if image is None or image.size == 0:
            # Log warning and return a default black image
            print(f"Warning: Failed to load image: {img_path}")
            image = np.zeros((160, 160, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        if self.transform:
            try:
                augmented = self.transform(image=image)
                image = augmented["image"]
            except Exception as e:
                print(f"Warning: Augmentation failed for {img_path}: {e}")
                # Fallback to simple resize
                image = cv2.resize(image, (160, 160))
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            # Default: resize and to tensor
            image = cv2.resize(image, (160, 160))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.return_path:
            return image, class_idx, str(img_path)

        return image, class_idx

    def get_class_name(self, idx: int) -> str:
        """Get class name dari index."""
        return self.idx_to_class[idx]

    def get_class_samples(self, class_idx: int) -> List[Tuple[Path, int]]:
        """Get all samples untuk specific class."""
        return [(path, idx) for path, idx in self.samples if idx == class_idx]


class TripletDataset(Dataset):
    """
    Dataset untuk triplet loss training.

    Generate triplets (anchor, positive, negative) on-the-fly.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[A.Compose] = None,
        samples_per_class: int = 2,
    ):
        """
        Initialize triplet dataset.

        Args:
            root_dir: Root directory
            transform: Augmentation pipeline
            samples_per_class: Number of samples per class per epoch
        """
        self.base_dataset = FaceRecognitionDataset(root_dir, transform)
        self.samples_per_class = samples_per_class
        self.logger = setup_logger("TripletDataset")

        # Organize samples by class
        self.class_samples = {}
        for img_path, class_idx in self.base_dataset.samples:
            if class_idx not in self.class_samples:
                self.class_samples[class_idx] = []
            self.class_samples[class_idx].append((img_path, class_idx))

        self.num_classes = len(self.class_samples)
        self.logger.info(f"Triplet dataset: {self.num_classes} classes")

    def __len__(self) -> int:
        """Return effective dataset size."""
        return self.num_classes * self.samples_per_class

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get triplet (anchor, positive, negative).

        Returns:
            Tuple of (anchor, positive, negative) tensors
        """
        # Select anchor class
        anchor_class = idx % self.num_classes

        # Sample anchor and positive dari same class
        if len(self.class_samples[anchor_class]) < 2:
            # If only 1 sample, use same image for anchor and positive
            anchor_path = self.class_samples[anchor_class][0][0]
            positive_path = anchor_path
        else:
            anchor_path, positive_path = np.random.choice(
                [p for p, _ in self.class_samples[anchor_class]], size=2, replace=False
            )

        # Sample negative dari different class
        negative_class = np.random.choice(
            [c for c in self.class_samples.keys() if c != anchor_class]
        )
        negative_path = np.random.choice(
            [p for p, _ in self.class_samples[negative_class]]
        )

        # Load images
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)

        return anchor, positive, negative

    def _load_image(self, img_path: Path) -> torch.Tensor:
        """Load and transform image."""
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.base_dataset.transform:
            augmented = self.base_dataset.transform(image=image)
            image = augmented["image"]
        else:
            image = cv2.resize(image, (160, 160))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image


def create_dataloaders(
    config: dict, batch_size: Optional[int] = None, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, dan test dataloaders.

    Args:
        config: Configuration dictionary
        batch_size: Override batch size dari config
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> config = load_config()
        >>> train_loader, val_loader, test_loader = create_dataloaders(config)
    """
    logger = setup_logger("DataLoader")

    # Get paths
    train_dir = config["paths"]["train_dir"]
    val_dir = config["paths"]["val_dir"]
    test_dir = config["paths"]["test_dir"]

    # Get batch size
    if batch_size is None:
        batch_size = config["cnn_model"]["training"]["batch_size"]

    # Import augmentation
    from src.preprocessing.augmentation import create_augmentation_pipeline

    # Create transforms
    train_transform = create_augmentation_pipeline(
        mode="train", target_size=config["cnn_model"]["input_size"][0]
    )
    val_transform = create_augmentation_pipeline(
        mode="val", target_size=config["cnn_model"]["input_size"][0]
    )

    # Create datasets
    train_dataset = FaceRecognitionDataset(train_dir, transform=train_transform)
    
    # Check if validation dir exists and has images
    val_dataset = None
    if Path(val_dir).exists() and any(Path(val_dir).rglob("*.[jJ][pP][gG]")) or any(Path(val_dir).rglob("*.[pP][nN][gG]")):
        try:
            val_dataset = FaceRecognitionDataset(val_dir, transform=val_transform)
        except RuntimeError:
            logger.warning(f"Validation directory exists but no valid images found. Skipping validation.")
    else:
        logger.warning(f"No validation set found. Training without validation.")
    
    test_dataset = FaceRecognitionDataset(test_dir, transform=val_transform)

    logger.info(f"Train dataset: {len(train_dataset)} images")
    if val_dataset is not None:
        logger.info(f"Val dataset: {len(val_dataset)} images")
    else:
        logger.info(f"Val dataset: 0 images (skipped)")
    logger.info(f"Test dataset: {len(test_dataset)} images")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Train batches: {len(train_loader)}")
    if val_loader is not None:
        logger.info(f"Val batches: {len(val_loader)}")
    else:
        logger.info(f"Val batches: 0 (skipped)")
    logger.info(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test dataset loader."""
    from src.utils.config_loader import load_config

    print("Testing dataset loader...")

    # Load config
    config = load_config()

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Test batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Unique labels in batch: {len(torch.unique(labels))}")

    print("\n✓ Dataset loader test complete!")
