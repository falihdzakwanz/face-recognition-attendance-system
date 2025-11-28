"""
CNN Model - FaceNet Implementation
===================================

FaceNet model untuk face recognition menggunakan InceptionResNetV1/V2
dengan triplet loss atau classification loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from typing import Optional, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger


class FaceNetEmbedding(nn.Module):
    """
    FaceNet model untuk generate face embeddings.

    Architecture:
        Input (160x160x3) → InceptionResNetV1 → Embedding (128/512-dim)

    Attributes:
        backbone: InceptionResNetV1 backbone
        embedding_size: Size of embedding vector
        pretrained: Whether using pretrained weights
    """

    def __init__(
        self,
        embedding_size: int = 128,
        pretrained: str = "vggface2",
        dropout: float = 0.5,
        freeze_backbone: bool = True,
        trainable_layers: int = 20,
    ):
        """
        Initialize FaceNet model.

        Args:
            embedding_size: Size of face embedding (128 or 512)
            pretrained: Pretrained weights ('vggface2', 'casia-webface', or None)
            dropout: Dropout rate
            freeze_backbone: Freeze early layers
            trainable_layers: Number of layers to train from end
        """
        super(FaceNetEmbedding, self).__init__()

        self.embedding_size = embedding_size
        self.logger = setup_logger("FaceNet")

        # Load pretrained InceptionResNetV1
        self.backbone = InceptionResnetV1(
            pretrained=pretrained, classify=False, dropout_prob=dropout
        )

        self.logger.info(f"Loaded InceptionResNetV1 (pretrained: {pretrained})")

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_layers(trainable_layers)

        # Embedding layer (512 → embedding_size)
        if embedding_size != 512:
            self.embedding_layer = nn.Sequential(
                nn.Linear(512, embedding_size), nn.BatchNorm1d(embedding_size)
            )
        else:
            self.embedding_layer = None

        self.logger.info(f"Embedding size: {embedding_size}")

    def _freeze_layers(self, trainable_layers: int):
        """Freeze early layers, train only last N layers."""
        # Get all parameters
        params = list(self.backbone.parameters())
        total_params = len(params)

        # Freeze first (total - trainable) layers
        freeze_until = max(0, total_params - trainable_layers)

        for i, param in enumerate(params):
            if i < freeze_until:
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.backbone.parameters())

        self.logger.info(f"Frozen {freeze_until}/{total_params} layer groups")
        self.logger.info(
            f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, 160, 160)

        Returns:
            Embeddings (B, embedding_size)
        """
        # Get features from backbone
        features = self.backbone(x)  # (B, 512)

        # Project to embedding size if needed
        if self.embedding_layer is not None:
            embeddings = self.embedding_layer(features)
        else:
            embeddings = features

        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class FaceNetClassifier(nn.Module):
    """
    FaceNet dengan classification head.

    Untuk training dengan cross-entropy loss.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 128,
        pretrained: str = "vggface2",
        dropout: float = 0.5,
        freeze_backbone: bool = True,
        trainable_layers: int = 20,
    ):
        """
        Initialize FaceNet classifier.

        Args:
            num_classes: Number of classes (students)
            embedding_size: Size of embedding layer
            pretrained: Pretrained weights
            dropout: Dropout rate
            freeze_backbone: Freeze early layers
            trainable_layers: Number of layers to train
        """
        super(FaceNetClassifier, self).__init__()

        self.num_classes = num_classes
        self.logger = setup_logger("FaceNetClassifier")

        # Embedding model
        self.embedding = FaceNetEmbedding(
            embedding_size=embedding_size,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            trainable_layers=trainable_layers,
        )

        # Classification head
        self.classifier = nn.Linear(embedding_size, num_classes)

        self.logger.info(f"Number of classes: {num_classes}")

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input images
            return_embedding: Return embeddings along with logits

        Returns:
            Logits (B, num_classes) or (logits, embeddings)
        """
        # Get embeddings
        embeddings = self.embedding(x)

        # Classification
        logits = self.classifier(embeddings)

        if return_embedding:
            return logits, embeddings

        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get face embeddings only."""
        return self.embedding(x)


class ArcFaceLayer(nn.Module):
    """
    ArcFace loss layer untuk angular margin.

    Reference: https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ):
        """
        Initialize ArcFace layer.

        Args:
            embedding_size: Input embedding size
            num_classes: Number of classes
            scale: Scale factor (s)
            margin: Angular margin (m) in radians
        """
        super(ArcFaceLayer, self).__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        # Weight matrix (num_classes, embedding_size)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: Normalized embeddings (B, embedding_size)
            labels: Ground truth labels (B,)

        Returns:
            Logits with angular margin (B, num_classes)
        """
        # Normalize weights
        normalized_weights = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, normalized_weights)
        cosine = torch.clamp(cosine, -1.0, 1.0)

        # Get angle
        theta = torch.acos(cosine)

        # Add angular margin to target class
        target_logit = torch.cos(theta + self.margin)

        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), dtype=cosine.dtype, device=cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin to target class only
        output = (one_hot * target_logit) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class FaceNetArcFace(nn.Module):
    """
    FaceNet dengan ArcFace loss.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 128,
        pretrained: str = "vggface2",
        dropout: float = 0.5,
        freeze_backbone: bool = True,
        trainable_layers: int = 20,
        scale: float = 30.0,
        margin: float = 0.5,
    ):
        """Initialize FaceNet with ArcFace."""
        super(FaceNetArcFace, self).__init__()

        self.logger = setup_logger("FaceNetArcFace")

        # Embedding model
        self.embedding = FaceNetEmbedding(
            embedding_size=embedding_size,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            trainable_layers=trainable_layers,
        )

        # ArcFace layer
        self.arcface = ArcFaceLayer(
            embedding_size=embedding_size,
            num_classes=num_classes,
            scale=scale,
            margin=margin,
        )

        self.logger.info(f"ArcFace: scale={scale}, margin={margin}")

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images
            labels: Ground truth labels (required for training)

        Returns:
            (logits, embeddings)
        """
        # Get embeddings
        embeddings = self.embedding(x)

        # Apply ArcFace if labels provided (training mode)
        if labels is not None:
            logits = self.arcface(embeddings, labels)
        else:
            # Inference mode: compute cosine similarity consistent with ArcFace training
            # Normalize embeddings and class weights, then apply the same scaling used during training
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            weight_norm = F.normalize(self.arcface.weight, p=2, dim=1)
            logits = F.linear(embeddings_norm, weight_norm) * getattr(
                self.arcface, "scale", 1.0
            )

        return logits, embeddings


def create_model(
    config: dict, num_classes: int, model_type: str = "classifier"
) -> nn.Module:
    """
    Factory function untuk create model.

    Args:
        config: Configuration dictionary
        num_classes: Number of classes
        model_type: 'embedding', 'classifier', or 'arcface'

    Returns:
        Model instance

    Example:
        >>> config = load_config()
        >>> model = create_model(config, num_classes=70, model_type='arcface')
    """
    logger = setup_logger("ModelFactory")

    cnn_config = config["cnn_model"]

    if model_type == "embedding":
        model = FaceNetEmbedding(
            embedding_size=cnn_config["embedding_size"],
            pretrained="vggface2",
            dropout=cnn_config["dropout"],
            freeze_backbone=cnn_config["freeze_layers"],
            trainable_layers=cnn_config["trainable_layers"],
        )

    elif model_type == "classifier":
        model = FaceNetClassifier(
            num_classes=num_classes,
            embedding_size=cnn_config["embedding_size"],
            pretrained="vggface2",
            dropout=cnn_config["dropout"],
            freeze_backbone=cnn_config["freeze_layers"],
            trainable_layers=cnn_config["trainable_layers"],
        )

    elif model_type == "arcface":
        model = FaceNetArcFace(
            num_classes=num_classes,
            embedding_size=cnn_config["embedding_size"],
            pretrained="vggface2",
            dropout=cnn_config["dropout"],
            freeze_backbone=cnn_config["freeze_layers"],
            trainable_layers=cnn_config["trainable_layers"],
            scale=30.0,
            margin=0.5,
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
    """Test model."""
    from src.utils.config_loader import load_config

    print("Testing FaceNet models...")

    config = load_config()
    num_classes = 70

    # Test embedding model
    print("\n1. Testing FaceNet Embedding...")
    model_emb = create_model(config, num_classes, "embedding")
    x = torch.randn(4, 3, 160, 160)
    emb = model_emb(x)
    print(f"   Input: {x.shape} → Embedding: {emb.shape}")
    print(f"   Embedding norm: {torch.norm(emb, dim=1)}")  # Should be ~1.0

    # Test classifier
    print("\n2. Testing FaceNet Classifier...")
    model_clf = create_model(config, num_classes, "classifier")
    logits = model_clf(x)
    print(f"   Input: {x.shape} → Logits: {logits.shape}")

    # Test ArcFace
    print("\n3. Testing FaceNet ArcFace...")
    model_arc = create_model(config, num_classes, "arcface")
    labels = torch.randint(0, num_classes, (4,))
    logits, emb = model_arc(x, labels)
    print(f"   Input: {x.shape} → Logits: {logits.shape}, Embeddings: {emb.shape}")

    print("\n✓ Model test complete!")
