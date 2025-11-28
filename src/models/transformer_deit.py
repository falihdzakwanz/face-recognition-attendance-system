"""
Transformer Model - DeiT Implementation
========================================

Data-efficient Image Transformer (DeiT) untuk face recognition.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger


class DeiTFaceRecognition(nn.Module):
    """
    DeiT model untuk face recognition.

    Architecture:
        Input (224x224x3) → DeiT → Classification Head → num_classes

    Attributes:
        backbone: DeiT backbone dari timm
        num_classes: Number of classes
        freeze_backbone: Whether to freeze backbone
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "deit_small_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        trainable_blocks: int = 2,
        dropout: float = 0.6,
    ):
        """
        Initialize DeiT model.

        Args:
            num_classes: Number of classes (students)
            model_name: Model name from timm
            pretrained: Use pretrained weights
            freeze_backbone: Freeze early layers
            trainable_blocks: Number of transformer blocks to train
            dropout: Dropout rate
        """
        super(DeiTFaceRecognition, self).__init__()

        self.num_classes = num_classes
        self.logger = setup_logger("DeiT")

        # Load pretrained DeiT
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_rate=dropout,
        )

        self.logger.info(f"Loaded {model_name} (pretrained: {pretrained})")

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_layers(trainable_blocks)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.feature_dim, num_classes)
        )

        self.logger.info(f"Number of classes: {num_classes}")
        self.logger.info(f"Feature dimension: {self.feature_dim}")

    def _freeze_layers(self, trainable_blocks: int):
        """
        Freeze early layers, train only last N transformer blocks.

        Args:
            trainable_blocks: Number of blocks to keep trainable
        """
        # Freeze patch embedding and positional embedding
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False

        if hasattr(self.backbone, "pos_embed"):
            self.backbone.pos_embed.requires_grad = False

        if hasattr(self.backbone, "cls_token"):
            self.backbone.cls_token.requires_grad = False

        # Freeze transformer blocks except last N
        if hasattr(self.backbone, "blocks"):
            total_blocks = len(self.backbone.blocks)
            freeze_until = max(0, total_blocks - trainable_blocks)

            for i, block in enumerate(self.backbone.blocks):
                for param in block.parameters():
                    param.requires_grad = i >= freeze_until

            self.logger.info(
                f"Frozen {freeze_until}/{total_blocks} transformer blocks, "
                f"training last {trainable_blocks} blocks"
            )

        # Count parameters
        trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.backbone.parameters())

        self.logger.info(
            f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)"
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, 224, 224)
            return_features: Return features along with logits

        Returns:
            Logits (B, num_classes) or (logits, features)
        """
        # Extract features
        features = self.backbone(x)  # (B, feature_dim)

        # Classification
        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features only."""
        return self.backbone(x)


class ViTFaceRecognition(nn.Module):
    """
    Vision Transformer (ViT) untuk face recognition.

    Alternative to DeiT dengan architecture serupa.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        trainable_blocks: int = 2,
        dropout: float = 0.6,
    ):
        """Initialize ViT model."""
        super(ViTFaceRecognition, self).__init__()

        self.num_classes = num_classes
        self.logger = setup_logger("ViT")

        # Load pretrained ViT
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, drop_rate=dropout
        )

        self.logger.info(f"Loaded {model_name} (pretrained: {pretrained})")

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_layers(trainable_blocks)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.feature_dim, num_classes)
        )

        self.logger.info(f"Number of classes: {num_classes}")
        self.logger.info(f"Feature dimension: {self.feature_dim}")

    def _freeze_layers(self, trainable_blocks: int):
        """Freeze early layers."""
        # Similar to DeiT
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False

        if hasattr(self.backbone, "pos_embed"):
            self.backbone.pos_embed.requires_grad = False

        if hasattr(self.backbone, "cls_token"):
            self.backbone.cls_token.requires_grad = False

        if hasattr(self.backbone, "blocks"):
            total_blocks = len(self.backbone.blocks)
            freeze_until = max(0, total_blocks - trainable_blocks)

            for i, block in enumerate(self.backbone.blocks):
                for param in block.parameters():
                    param.requires_grad = i >= freeze_until

            self.logger.info(f"Frozen {freeze_until}/{total_blocks} transformer blocks")

        trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.backbone.parameters())
        self.logger.info(
            f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)"
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features only."""
        return self.backbone(x)


def create_transformer_model(
    config: dict, num_classes: int, model_type: str = "deit"
) -> nn.Module:
    """
    Factory function untuk create transformer model.

    Args:
        config: Configuration dictionary
        num_classes: Number of classes
        model_type: 'deit', 'vit', or 'swin'

    Returns:
        Model instance
    """
    logger = setup_logger("TransformerFactory")

    transformer_config = config["transformer_model"]

    if model_type == "deit":
        model = DeiTFaceRecognition(
            num_classes=num_classes,
            model_name=transformer_config["architecture"],
            pretrained=transformer_config["pretrained"],
            freeze_backbone=transformer_config["freeze_backbone"],
            trainable_blocks=transformer_config["trainable_blocks"],
            dropout=transformer_config["dropout"],
        )

    elif model_type == "vit":
        model = ViTFaceRecognition(
            num_classes=num_classes,
            model_name="vit_base_patch16_224",
            pretrained=transformer_config["pretrained"],
            freeze_backbone=transformer_config["freeze_backbone"],
            trainable_blocks=transformer_config["trainable_blocks"],
            dropout=transformer_config["dropout"],
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model type: {model_type}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)"
    )

    return model


if __name__ == "__main__":
    """Test transformer models."""
    from src.utils.config_loader import load_config

    print("Testing Transformer models...")

    config = load_config()
    num_classes = 70

    # Test DeiT
    print("\n1. Testing DeiT...")
    model_deit = create_transformer_model(config, num_classes, "deit")
    x = torch.randn(4, 3, 224, 224)
    logits = model_deit(x)
    print(f"   Input: {x.shape} → Logits: {logits.shape}")

    # Test ViT
    print("\n2. Testing ViT...")
    model_vit = create_transformer_model(config, num_classes, "vit")
    logits = model_vit(x)
    print(f"   Input: {x.shape} → Logits: {logits.shape}")

    print("\n✓ Transformer model test complete!")
