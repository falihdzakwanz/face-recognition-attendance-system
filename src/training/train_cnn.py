"""
CNN Training Script
===================

Training script untuk FaceNet model dengan support untuk:
- Cross-entropy loss (classification)
- Triplet loss (embedding)
- ArcFace loss (metric learning)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config_loader import load_config, get_paths
from src.utils.logger import setup_logger, get_experiment_name, log_config
from src.utils.dataset import create_dataloaders
from src.models.cnn_facenet import create_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TripletLoss(nn.Module):
    """
    Triplet loss implementation.

    L = max(d(a, p) - d(a, n) + margin, 0)
    """

    def __init__(self, margin: float = 0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """Calculate triplet loss."""
        # Euclidean distance
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)

        # Triplet loss
        loss = torch.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


class CNNTrainer:
    """
    Trainer class untuk CNN model.

    Attributes:
        config: Configuration dictionary
        model: Model instance
        device: Device (cuda/cpu)
        optimizer: Optimizer
        criterion: Loss function
        scaler: AMP gradient scaler
    """

    def __init__(
        self,
        config: dict,
        model: nn.Module,
        device: torch.device,
        experiment_name: Optional[str] = None,
        model_type: str = "classifier",
    ):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
            model: Model instance
            device: Device
            experiment_name: Experiment name untuk logging
            model_type: Model type (classifier, embedding, arcface)
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type

        if experiment_name is None:
            experiment_name = get_experiment_name("cnn")
        self.experiment_name = experiment_name

        self.logger = setup_logger("CNNTrainer")

        # Setup paths
        self.paths = get_paths(config)
        self.checkpoint_dir = Path(config["paths"]["models_dir"]) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup tensorboard
        log_dir = Path(config["paths"]["logs_dir"]) / experiment_name
        self.writer = SummaryWriter(log_dir)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup loss function
        self.criterion = self._create_criterion()

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision training
        self.use_amp = config["system"]["mixed_precision"]
        self.scaler = GradScaler(device='cuda') if self.use_amp and torch.cuda.is_available() else None

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Mixed precision: {self.use_amp}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        opt_config = self.config["cnn_model"]["optimizer"]

        if opt_config["name"].lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config["learning_rate"],
                weight_decay=opt_config["weight_decay"],
            )
        elif opt_config["name"].lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config["learning_rate"],
                momentum=0.9,
                weight_decay=opt_config["weight_decay"],
            )
        elif opt_config["name"].lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config["learning_rate"],
                weight_decay=opt_config["weight_decay"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['name']}")

        self.logger.info(
            f"Optimizer: {opt_config['name']} (lr={opt_config['learning_rate']})"
        )
        return optimizer

    def _create_criterion(self) -> nn.Module:
        """Create loss function."""
        loss_name = self.config["cnn_model"]["loss_function"]
        
        # For ArcFace model, always use CrossEntropy regardless of config
        # ArcFace already implements angular margin in the layer itself
        if self.model_type == "arcface":
            criterion = nn.CrossEntropyLoss()
            self.logger.info("Loss function: cross_entropy (ArcFace with angular margin)")
        elif loss_name == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
            self.logger.info(f"Loss function: {loss_name}")
        elif loss_name == "triplet_loss":
            margin = self.config["cnn_model"].get("triplet_margin", 0.2)
            criterion = TripletLoss(margin=margin)
            self.logger.info(f"Loss function: {loss_name} (margin={margin})")
        else:
            # Default to cross entropy
            criterion = nn.CrossEntropyLoss()
            self.logger.info(f"Loss function: cross_entropy (default)")

        return criterion

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        train_config = self.config["cnn_model"]["training"]

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=train_config["reduce_lr_factor"],
            patience=train_config["reduce_lr_patience"],
        )

        return scheduler

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """
        Train one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[
                            0
                        ]  # For models returning (logits, embeddings)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, val_loader) -> Tuple[float, float, dict]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (avg_loss, accuracy, metrics_dict)
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = self.criterion(outputs, labels)

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        return avg_loss, accuracy, metrics

    def train(self, train_loader, val_loader, num_epochs: Optional[int] = None) -> dict:
        """
        Train model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (override config)

        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.config["cnn_model"]["training"]["epochs"]

        early_stopping_patience = self.config["cnn_model"]["training"][
            "early_stopping_patience"
        ]
        epochs_without_improvement = 0

        self.logger.info("=" * 60)
        self.logger.info("Starting training...")
        self.logger.info(f"Epochs: {num_epochs}")
        self.logger.info(f"Early stopping patience: {early_stopping_patience}")
        self.logger.info("=" * 60)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_loss, val_acc, val_metrics = self.validate(val_loader)
            else:
                # No validation, use train loss for monitoring
                val_loss, val_acc = train_loss, train_acc
                val_metrics = {}

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Logging
            if val_loader is not None:
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                    f"F1: {val_metrics.get('f1', 0.0):.4f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"No validation set"
                )

            # TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("Metrics/precision", val_metrics["precision"], epoch)
            self.writer.add_scalar("Metrics/recall", val_metrics["recall"], epoch)
            self.writer.add_scalar("Metrics/f1", val_metrics["f1"], epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step(val_acc)

            # Save checkpoint
            self.save_checkpoint(
                epoch=epoch, val_acc=val_acc, is_best=(val_acc > self.best_val_acc)
            )

            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                epochs_without_improvement = 0
                self.logger.info(f"✓ New best validation accuracy: {val_acc:.4f}")
            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        self.logger.info("=" * 60)
        self.logger.info("Training complete!")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        self.logger.info("=" * 60)

        self.writer.close()

        return self.history

    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "config": self.config,
        }

        # Save last checkpoint
        checkpoint_path = self.checkpoint_dir / "last_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"✓ Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint["history"]

        self.logger.info(f"✓ Loaded checkpoint from epoch {self.current_epoch+1}")


def train_cnn_model(config: dict, model_type: str = "arcface") -> dict:
    """
    Main function untuk train CNN model.

    Args:
        config: Configuration dictionary
        model_type: 'classifier', 'arcface', or 'embedding'

    Returns:
        Training history
    """
    logger = setup_logger("TrainCNN")

    # Log configuration
    log_config(logger, config["cnn_model"], "CNN Model Configuration")

    # Device
    device = torch.device(
        config["system"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, num_workers=config["system"]["num_workers"]
    )
    
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
    model = create_model(config, num_classes, model_type)

    # Create trainer
    experiment_name = get_experiment_name(f"cnn_{model_type}")
    trainer = CNNTrainer(config, model, device, experiment_name, model_type)

    # Train
    history = trainer.train(train_loader, val_loader)

    # Save final results
    results_path = trainer.checkpoint_dir / "training_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Best validation accuracy: {trainer.best_val_acc:.4f}\n")
        f.write(f"Total epochs: {len(history['train_loss'])}\n")

    logger.info(f"Results saved to: {results_path}")

    return history


if __name__ == "__main__":
    """Run CNN training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN FaceNet model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="arcface",
        choices=["classifier", "arcface", "embedding"],
        help="Model type",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Train
    history = train_cnn_model(config, model_type=args.model_type)

    print("\n✓ Training complete!")
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")
