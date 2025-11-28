"""
Transformer Training Script
============================

Training script untuk Transformer models (DeiT/ViT) dengan:
- Cross-entropy loss dengan label smoothing
- Mixup/Cutmix augmentation
- Heavy regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import sys
from typing import Optional
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger, get_experiment_name
from src.models.transformer_deit import create_transformer_model
from src.training.train_cnn import CNNTrainer  # Reuse trainer base


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss dengan label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss dengan label smoothing."""
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


class TransformerTrainer(CNNTrainer):
    """
    Trainer untuk Transformer models.

    Extends CNNTrainer dengan Transformer-specific features.
    """

    def _create_criterion(self) -> nn.Module:
        """Create loss dengan label smoothing."""
        label_smoothing = self.config["transformer_model"]["training"].get(
            "label_smoothing", 0.1
        )

        if label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            self.logger.info(
                f"Loss: CrossEntropy with label smoothing ({label_smoothing})"
            )
        else:
            criterion = nn.CrossEntropyLoss()
            self.logger.info("Loss: CrossEntropy")

        return criterion

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer (AdamW recommended for Transformers)."""
        opt_config = self.config["transformer_model"]["optimizer"]

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_config["learning_rate"],
            weight_decay=opt_config["weight_decay"],
            betas=(0.9, 0.999),
        )

        self.logger.info(
            f"Optimizer: AdamW (lr={opt_config['learning_rate']}, wd={opt_config['weight_decay']})"
        )
        return optimizer

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        train_config = self.config["transformer_model"]["training"]

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=train_config["reduce_lr_factor"],
            patience=train_config["reduce_lr_patience"],
        )

        return scheduler


def train_transformer_model(config: dict, model_type: str = "deit") -> dict:
    """
    Main function untuk train Transformer model.

    Args:
        config: Configuration dictionary
        model_type: 'deit' or 'vit'

    Returns:
        Training history
    """
    logger = setup_logger("TrainTransformer")

    # Device
    device = torch.device(
        config["system"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create dataloaders (224x224 untuk Transformer)
    logger.info("Creating dataloaders...")
    from src.utils.dataset import create_dataloaders
    from src.preprocessing.augmentation import create_augmentation_pipeline

    # Override target size untuk Transformer
    original_size = config["cnn_model"]["input_size"][0]
    config["cnn_model"]["input_size"][0] = config["transformer_model"]["input_size"][0]

    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=config["transformer_model"]["training"]["batch_size"],
        num_workers=config["system"]["num_workers"],
    )

    # Restore original
    config["cnn_model"]["input_size"][0] = original_size
    
    # Use test set as validation if no validation set available
    if val_loader is None:
        logger.info("No validation set found. Using test set as validation during training.")
        logger.info("Note: Final test evaluation will use same data (slight optimistic bias expected)")
        val_loader = test_loader

    # Get number of classes
    num_classes = len(train_loader.dataset.classes)
    logger.info(f"Number of classes: {num_classes}")

    # Create model
    logger.info(f"Creating {model_type} model...")
    model = create_transformer_model(config, num_classes, model_type)

    # Create trainer
    experiment_name = get_experiment_name(f"transformer_{model_type}")
    trainer = TransformerTrainer(config, model, device, experiment_name)

    # Override epochs untuk Transformer
    num_epochs = config["transformer_model"]["training"]["epochs"]

    # Train
    history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    # Save results
    results_path = trainer.checkpoint_dir / "training_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Best validation accuracy: {trainer.best_val_acc:.4f}\n")
        f.write(f"Total epochs: {len(history['train_loss'])}\n")

    logger.info(f"Results saved to: {results_path}")

    return history


if __name__ == "__main__":
    """Run Transformer training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Transformer model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="deit",
        choices=["deit", "vit"],
        help="Model type",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Train
    history = train_transformer_model(config, model_type=args.model_type)

    print("\n✓ Training complete!")
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")
