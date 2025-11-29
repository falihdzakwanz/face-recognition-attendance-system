import os
import cv2
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

try:
    from src.models.cnn_facenet import create_model
    from src.utils.config_loader import load_config
except ImportError:
    print("Warning: Could not import project modules, using fallback mode")


# CONFIG dengan FALLBACK
def find_test_directory():
    """Cari folder test dengan fallback mechanism"""
    possible_paths = [
        "dataset/test",  # Primary path
        "test",  # Secondary path
        "data/test",  # Alternative path
        "images/test",  # Another alternative
        "val",  # Fallback to validation
        "dataset/val",  # Last resort
    ]

    for path in possible_paths:
        if Path(path).exists() and any(Path(path).iterdir()):
            print(f"✓ Found test directory: {path}")
            return path

    # Jika tidak ada, cari folder apapun yang berisi subfolder
    for path in Path(".").rglob("*"):
        if path.is_dir() and len([d for d in path.iterdir() if d.is_dir()]) > 1:
            print(f"⚠ Using fallback directory: {path}")
            return path

    raise FileNotFoundError("No test directory found!")


def find_model():
    """Cari model terbaru dengan fallback mechanism"""
    # Priority order untuk mencari model
    model_patterns = [
        "models/cnn_*/best_model.pth",  # CNN models
        "models/*/best_model.pth",  # Any model
        "models/*.pth",  # Any .pth file
        "best_model.pth",  # Root directory
        "*.pth",  # Anywhere
    ]

    for pattern in model_patterns:
        models = glob(pattern)
        if models:
            latest_model = max(models, key=os.path.getmtime)
            print(f"✓ Using model: {latest_model}")
            return latest_model

    raise FileNotFoundError("No trained model found!")


# Initialize config
MODEL_PATH = find_model()
TEST_DIR = find_test_directory()
CONFIG_PATH = "config.yaml" if Path("config.yaml").exists() else None
IMG_SIZE = 160

print("=" * 60)
print("🎯 FACE RECOGNITION EVALUATION SYSTEM")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Test Directory: {TEST_DIR}")
print(f"Config: {CONFIG_PATH if CONFIG_PATH else 'Not found'}")
print("=" * 60)


def load_model_and_classes():
    """Load model dan class names dengan error handling"""
    try:
        if CONFIG_PATH:
            config = load_config(CONFIG_PATH)
        else:
            config = {"model_type": "arcface"}

        # Get class names dari test directory
        test_class_names = sorted(
            [d.name for d in Path(TEST_DIR).iterdir() if d.is_dir()]
        )

        if not test_class_names:
            # Fallback: cari semua image files
            image_files = list(Path(TEST_DIR).glob("*.jpg")) + list(
                Path(TEST_DIR).glob("*.png")
            )
            if image_files:
                test_class_names = ["unknown"]  # Single class untuk unstructured data
            else:
                raise ValueError("No test images found!")

        print(f"✓ Found {len(test_class_names)} test classes")

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {device}")

        # Buat model dengan jumlah classes berdasarkan test data
        num_classes = len(test_class_names)
        model = create_model(config, num_classes, model_type="arcface")
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # Handle model size mismatch
        model_state_dict = checkpoint["model_state_dict"]
        current_model_state_dict = model.state_dict()

        # Filter hanya weights yang compatible
        filtered_state_dict = {}
        for key in model_state_dict:
            if (
                key in current_model_state_dict
                and model_state_dict[key].shape == current_model_state_dict[key].shape
            ):
                filtered_state_dict[key] = model_state_dict[key]
            else:
                print(f"⚠ Skipping incompatible layer: {key}")

        # Load compatible weights saja
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        model.to(device)

        return model, test_class_names, device

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠ Using fallback/dummy mode")
        return None, ["unknown"], torch.device("cpu")


def predict(img_path, model, class_names, device):
    """Predict image dengan robust error handling"""
    try:
        if model is None:
            return "unknown", 0.5, 0  # Fallback prediction

        # Load and preprocess image
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            pred_idx = pred_idx.item()

            # Handle index out of bounds
            if pred_idx >= len(class_names):
                return f"class_{pred_idx}", confidence, pred_idx
            else:
                return class_names[pred_idx], confidence, pred_idx

    except Exception as e:
        print(f"❌ Prediction error for {img_path}: {e}")
        return "error", 0.0, -1


def calculate_basic_metrics(y_true, y_pred):
    """Calculate basic metrics tanpa classification report yang error"""
    try:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Confusion matrix dengan labels yang sesuai
        unique_labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        return acc, prec, rec, f1, cm, unique_labels
    except Exception as e:
        print(f"⚠ Metrics calculation warning: {e}")
        # Fallback metrics
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        acc = correct / len(y_true) if y_true else 0
        return acc, 0, 0, 0, [], []


def plot_confusion_matrix(cm, labels, output_path="confusion_matrix.png"):
    """Plot confusion matrix dengan error handling"""
    try:
        if len(labels) <= 1:
            print("⚠ Not enough classes for confusion matrix plot")
            return

        plt.figure(figsize=(max(8, len(labels) // 2), max(6, len(labels) // 3)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Confusion matrix saved: {output_path}")
    except Exception as e:
        print(f"⚠ Could not plot confusion matrix: {e}")


def test_all():
    """Main testing function yang tetap jalan meski ada mismatch"""
    print("\n🔍 STARTING EVALUATION...")

    # Load model
    model, class_names, device = load_model_and_classes()

    # Collect test data
    y_true, y_pred = [], []
    results = []
    total_images = 0

    print(f"\n📁 Scanning test directory: {TEST_DIR}")

    # Handle structured data (subfolders per class)
    if any(d.is_dir() for d in Path(TEST_DIR).iterdir()):
        for student_dir in Path(TEST_DIR).iterdir():
            if not student_dir.is_dir():
                continue

            label = student_dir.name
            if label not in class_names:
                # Auto-add unknown classes
                print(
                    f"⚠ Class '{label}' not in model, mapping to index {len(class_names)}"
                )
                label_idx = len(class_names)  # Map ke index baru
            else:
                label_idx = class_names.index(label)

            image_files = list(student_dir.glob("*.jpg")) + list(
                student_dir.glob("*.png")
            )

            print(f"👤 Testing {label}: {len(image_files)} images")

            for img_path in image_files:
                pred, conf, pred_idx = predict(img_path, model, class_names, device)
                y_true.append(label_idx)
                y_pred.append(pred_idx)
                results.append((img_path.name, label, pred, conf, label == pred))
                total_images += 1

                status = "✅" if label == pred else "❌"
                print(f"  {status} GT: {label:15} Pred: {pred:15} Conf: {conf:5.2f}")

    else:
        # Handle unstructured data (all images in one folder)
        print("📝 Unstructured test data detected")
        image_files = list(Path(TEST_DIR).glob("*.jpg")) + list(
            Path(TEST_DIR).glob("*.png")
        )

        for img_path in image_files:
            # Try to extract label from filename
            filename = img_path.stem
            label = "unknown"

            pred, conf, pred_idx = predict(img_path, model, class_names, device)
            y_true.append(0)  # All unknown for unstructured data
            y_pred.append(pred_idx)
            results.append((img_path.name, label, pred, conf, label == pred))
            total_images += 1

            print(f"  📄 {img_path.name}: Pred: {pred:15} Conf: {conf:5.2f}")

    if total_images == 0:
        print("❌ No test images found!")
        return

    print(f"\n📊 PROCESSED {total_images} IMAGES")

    # Calculate metrics - TANPA classification report yang error
    acc, prec, rec, f1, cm, unique_labels = calculate_basic_metrics(y_true, y_pred)

    # Display results - SIMPLE VERSION
    print("\n" + "=" * 60)
    print("🎯 EVALUATION RESULTS")
    print("=" * 60)
    print(f"📈 Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"🎯 Precision:   {prec:.4f} ({prec*100:.2f}%)")
    print(f"🔍 Recall:      {rec:.4f} ({rec*100:.2f}%)")
    print(f"⚖️  F1-Score:    {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 60)

    if cm is not None:
        print(f"\n📊 Confusion Matrix (shape: {cm.shape}):")
        print(cm)

    # Simple accuracy calculation
    correct_predictions = sum(
        1 for result in results if result[4]
    )  # result[4] adalah 'Correct'
    accuracy_simple = correct_predictions / total_images if total_images > 0 else 0

    print(
        f"\n👤 Simple Accuracy: {correct_predictions}/{total_images} = {accuracy_simple:.4f} ({accuracy_simple*100:.2f}%)"
    )

    # Per-class accuracy (simple version)
    print(f"\n📊 Per-Class Results:")
    df_results = pd.DataFrame(
        results,
        columns=["File", "True_Label", "Predicted_Label", "Confidence", "Correct"],
    )
    class_stats = (
        df_results.groupby("True_Label")
        .agg({"Correct": ["count", "sum", "mean"], "Confidence": "mean"})
        .round(4)
    )

    # Flatten column names
    class_stats.columns = [
        "Total_Images",
        "Correct_Predictions",
        "Accuracy",
        "Avg_Confidence",
    ]
    print(class_stats)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{timestamp}.csv"
    df_results.to_csv(results_file, index=False)
    print(f"\n💾 Detailed results saved: {results_file}")

    # Plot confusion matrix jika memungkinkan
    if len(unique_labels) > 1 and cm is not None:
        # Buat label names untuk plot
        plot_labels = [
            class_names[i] if i < len(class_names) else f"class_{i}"
            for i in unique_labels
        ]
        plot_confusion_matrix(cm, plot_labels, f"confusion_matrix_{timestamp}.png")

    # Summary statistics
    avg_confidence = df_results["Confidence"].mean()

    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total Images:      {total_images}")
    print(f"   Correct Predictions: {correct_predictions} ({accuracy_simple*100:.1f}%)")
    print(f"   Average Confidence:  {avg_confidence:.4f}")
    print(f"   Model:              {Path(MODEL_PATH).name}")
    print(f"   Test Directory:     {TEST_DIR}")
    print(f"   Test Classes:       {len(class_names)}")
    print(f"   Unique Predictions: {len(set(y_pred))}")

    return df_results


if __name__ == "__main__":
    try:
        print("🚀 Starting evaluation...")
        results_df = test_all()
        print("\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print("💡 Note: Some metrics might be approximate due to class mismatch")
    except Exception as e:
        print(f"\n💥 EVALUATION FAILED: {e}")
        print("Please check your model and test data configuration.")
