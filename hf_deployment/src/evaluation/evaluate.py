"""
Model Evaluation Module
========================

Comprehensive evaluation untuk CNN dan Transformer models dengan:
- Confusion matrix
- Per-class metrics
- Model comparison
- Inference speed testing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import time
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_model_comparison,
)
from src.utils.dataset import create_dataloaders


class ModelEvaluator:
    """
    Evaluator untuk trained models.

    Attributes:
        model: Trained model
        device: Device (cuda/cpu)
        class_names: List of class names
    """

    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str]):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            device: Device
            class_names: List of class names (student names)
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.logger = setup_logger("Evaluator")

        self.model.eval()

    @torch.no_grad()
    def predict(
        self, dataloader: DataLoader, return_probs: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions untuk entire dataset.

        Args:
            dataloader: Data loader
            return_probs: Return prediction probabilities

        Returns:
            Tuple of (predictions, labels, [probabilities])
        """
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(self.device)

            outputs = self.model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # For models returning (logits, features)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

            if return_probs:
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        if return_probs:
            all_probs = np.array(all_probs)
            return all_preds, all_labels, all_probs

        return all_preds, all_labels, None

    def evaluate(self, dataloader: DataLoader, save_dir: Optional[str] = None) -> Dict:
        """
        Comprehensive evaluation.

        Args:
            dataloader: Data loader
            save_dir: Directory untuk save results

        Returns:
            Dictionary dengan metrics
        """
        self.logger.info("Starting evaluation...")

        # Get predictions
        preds, labels, probs = self.predict(dataloader, return_probs=True)

        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = (
            precision_recall_fscore_support(
                labels, preds, average=None, zero_division=0
            )
        )

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        # Top-k accuracy
        top5_acc = self._top_k_accuracy(probs, labels, k=5)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "top5_accuracy": float(top5_acc),
            "per_class_metrics": {
                self.class_names[i]: {
                    "precision": float(per_class_precision[i]),
                    "recall": float(per_class_recall[i]),
                    "f1": float(per_class_f1[i]),
                }
                for i in range(len(self.class_names))
            },
        }

        # Logging
        self.logger.info("=" * 60)
        self.logger.info("Evaluation Results")
        self.logger.info("=" * 60)
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1-Score: {f1:.4f}")
        self.logger.info(f"Top-5 Accuracy: {top5_acc:.4f}")
        self.logger.info("=" * 60)

        # Save results
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save metrics JSON
            with open(save_path / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            # Save confusion matrix plot
            plot_confusion_matrix(
                labels,
                preds,
                self.class_names,
                save_path=str(save_path / "confusion_matrix.png"),
            )

            # Save per-class accuracy
            plot_per_class_accuracy(
                labels,
                preds,
                self.class_names,
                save_path=str(save_path / "per_class_accuracy.png"),
                top_n=15,
            )

            # Save classification report
            report = classification_report(
                labels, preds, target_names=self.class_names, zero_division=0
            )
            with open(save_path / "classification_report.txt", "w") as f:
                f.write(report)

            self.logger.info(f"Results saved to: {save_path}")

        return metrics

    def _top_k_accuracy(
        self, probs: np.ndarray, labels: np.ndarray, k: int = 5
    ) -> float:
        """Calculate top-k accuracy."""
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = np.any(top_k_preds == labels[:, np.newaxis], axis=1)
        return np.mean(correct)

    def measure_inference_speed(
        self,
        input_size: Tuple[int, int, int] = (3, 160, 160),
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Measure inference speed.

        Args:
            input_size: Input size (C, H, W)
            num_iterations: Number of iterations untuk averaging

        Returns:
            Dictionary dengan timing metrics
        """
        self.logger.info("Measuring inference speed...")

        # Create dummy input
        dummy_input = torch.randn(1, *input_size).to(self.device)

        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)

        # Measure
        times = []
        for _ in tqdm(range(num_iterations), desc="Timing"):
            start = time.time()
            with torch.no_grad():
                _ = self.model(dummy_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms

        metrics = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": float(1000 / np.mean(times)),
        }

        self.logger.info(
            f"Inference time: {metrics['mean_ms']:.2f} ± {metrics['std_ms']:.2f} ms"
        )
        self.logger.info(f"FPS: {metrics['fps']:.1f}")

        return metrics


def load_trained_model(
    checkpoint_path: str,
    model_class,
    config: dict,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Load trained model dari checkpoint.

    Args:
        checkpoint_path: Path ke checkpoint file
        model_class: Model class
        config: Configuration
        num_classes: Number of classes
        device: Device

    Returns:
        Loaded model
    """
    logger = setup_logger("ModelLoader")

    # Create model
    from src.models.cnn_facenet import create_model
    from src.models.transformer_deit import create_transformer_model

    if (
        "cnn" in str(checkpoint_path).lower()
        or "facenet" in str(checkpoint_path).lower()
    ):
        model = create_model(config, num_classes, model_type="arcface")
    else:
        model = create_transformer_model(config, num_classes, model_type="deit")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"✓ Loaded model from: {checkpoint_path}")
    logger.info(f"   Epoch: {checkpoint['epoch']+1}")
    logger.info(f"   Val Accuracy: {checkpoint['val_acc']:.4f}")

    return model


def compare_models(
    models_dict: Dict[str, Dict], config: dict, test_loader: DataLoader
) -> Dict:
    """
    Compare multiple models.

    Args:
        models_dict: Dictionary dengan format {name: {'model': model, 'checkpoint': path}}
        config: Configuration
        test_loader: Test data loader

    Returns:
        Comparison results
    """
    logger = setup_logger("ModelComparison")

    logger.info("=" * 60)
    logger.info("Comparing Models")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = test_loader.dataset.classes

    results = {}

    for name, info in models_dict.items():
        logger.info(f"\nEvaluating: {name}")
        logger.info("-" * 60)

        # Load model
        model = info["model"].to(device)

        # Evaluate
        evaluator = ModelEvaluator(model, device, class_names)
        metrics = evaluator.evaluate(test_loader)

        # Measure speed
        input_size = (3, 160, 160) if "cnn" in name.lower() else (3, 224, 224)
        speed_metrics = evaluator.measure_inference_speed(input_size)

        results[name] = {
            **metrics,
            "inference_time_ms": speed_metrics["mean_ms"],
            "fps": speed_metrics["fps"],
        }

    # Save comparison
    output_dir = Path(config["paths"]["output_dir"]) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot comparison
    comparison_metrics = {
        name: {
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1_score"],
        }
        for name, r in results.items()
    }

    plot_model_comparison(
        comparison_metrics, save_path=str(output_dir / "model_comparison.png")
    )

    # Save JSON
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info("\n" + "=" * 60)
    logger.info("Comparison Summary")
    logger.info("=" * 60)
    for name, metrics in results.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(
            f"  Inference: {metrics['inference_time_ms']:.2f} ms ({metrics['fps']:.1f} FPS)"
        )
    logger.info("=" * 60)

    return results


def evaluate_models(config: dict) -> Dict[str, dict]:
    """
    Main function untuk evaluasi dan comparison kedua model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary berisi hasil evaluasi untuk setiap model
    """
    logger = setup_logger("Evaluate")
    
    logger.info("\n" + "=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Load dataloaders
    logger.info("Loading test dataset...")
    _, _, test_loader = create_dataloaders(
        config, num_workers=config["system"]["num_workers"]
    )
    
    if test_loader is None or len(test_loader) == 0:
        logger.error("No test data available!")
        return {}
    
    device = torch.device(
        config["system"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")
    
    # Get class names
    class_names = test_loader.dataset.classes
    logger.info(f"Number of classes: {len(class_names)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    results = {}
    
    # Find latest trained models
    models_dir = Path(config["paths"]["models_dir"])
    
    # Find CNN model
    cnn_checkpoints = list(models_dir.glob("cnn_*/best_model.pth"))
    if cnn_checkpoints:
        latest_cnn = max(cnn_checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"\nEvaluating CNN: {latest_cnn.parent.name}")
        
        try:
            cnn_model = load_trained_model(
                str(latest_cnn),
                None,  # model_class (unused, auto-detected)
                config,
                len(class_names),
                device
            )
            
            cnn_evaluator = ModelEvaluator(cnn_model, device, class_names)
            cnn_results = cnn_evaluator.evaluate(test_loader)
            results["CNN (FaceNet + ArcFace)"] = cnn_results
            
            logger.info(f"CNN Accuracy: {cnn_results['accuracy']:.4f}")
            logger.info(f"CNN F1-Score: {cnn_results['f1_score']:.4f}")
            
        except Exception as e:
            logger.error(f"CNN evaluation failed: {e}")
    else:
        logger.warning("No CNN model found!")
    
    # Find Transformer model
    transformer_checkpoints = list(models_dir.glob("transformer_*/best_model.pth"))
    if transformer_checkpoints:
        latest_transformer = max(transformer_checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"\nEvaluating Transformer: {latest_transformer.parent.name}")
        
        try:
            from src.models.transformer_deit import create_transformer_model
            
            transformer_model = create_transformer_model(
                config,
                len(class_names),
                model_type="deit"
            )
            
            # Load checkpoint
            checkpoint = torch.load(latest_transformer, map_location=device)
            transformer_model.load_state_dict(checkpoint["model_state_dict"])
            
            transformer_evaluator = ModelEvaluator(transformer_model, device, class_names)
            transformer_results = transformer_evaluator.evaluate(test_loader)
            results["Transformer (DeiT)"] = transformer_results
            
            logger.info(f"Transformer Accuracy: {transformer_results['accuracy']:.4f}")
            logger.info(f"Transformer F1-Score: {transformer_results['f1_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Transformer evaluation failed: {e}")
    else:
        logger.warning("No Transformer model found!")
    
    # Compare models if both available
    if len(results) == 2:
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        
        for name, metrics in results.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if 'inference_time_ms' in metrics:
                logger.info(f"  Inference: {metrics['inference_time_ms']:.2f} ms ({metrics['fps']:.1f} FPS)")
        
        # Save comparison
        output_dir = Path(config["paths"]["output_dir"]) / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "comparison_results.json", "w") as f:
            json.dump(results, f, indent=4, default=str)
        
        logger.info(f"\nResults saved to: {output_dir}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Evaluation Complete")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    """Run evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument("--cnn-checkpoint", type=str, help="Path to CNN checkpoint")
    parser.add_argument(
        "--transformer-checkpoint", type=str, help="Path to Transformer checkpoint"
    )
    parser.add_argument("--compare", action="store_true", help="Compare both models")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create test loader
    _, _, test_loader = create_dataloaders(config)

    print("\n✓ Evaluation complete!")
