"""
Visualization Utilities
=======================

Module untuk plotting dan visualisasi hasil training & evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """
    Plot training history (loss & accuracy curves).
    
    Args:
        history: Dictionary dengan keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path untuk save plot (optional)
        figsize: Figure size
        
    Example:
        >>> history = {
        ...     'train_loss': [0.5, 0.4, 0.3],
        ...     'val_loss': [0.6, 0.5, 0.4],
        ...     'train_acc': [0.8, 0.85, 0.9],
        ...     'val_acc': [0.75, 0.8, 0.85]
        ... }
        >>> plot_training_history(history, save_path='outputs/training_history.png')
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 14),
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix dengan heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List nama kelas (mahasiswa)
        save_path: Path untuk save plot (optional)
        figsize: Figure size
        normalize: Normalize confusion matrix (percentage)
        
    Example:
        >>> y_true = np.array([0, 1, 2, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> class_names = ['Student A', 'Student B', 'Student C']
        >>> plot_confusion_matrix(y_true, y_pred, class_names)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=False,  # Too many classes to annotate
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.xticks(rotation=90, ha='right', fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Plot per-class accuracy dengan bar chart.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List nama kelas
        save_path: Path untuk save plot (optional)
        figsize: Figure size
        top_n: Show only top N and bottom N classes (optional)
        
    Returns:
        DataFrame berisi per-class accuracy
        
    Example:
        >>> df = plot_per_class_accuracy(y_true, y_pred, class_names, top_n=10)
    """
    # Compute per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Student': class_names,
        'Accuracy': per_class_acc,
        'Correct': cm.diagonal().astype(int),
        'Total': cm.sum(axis=1).astype(int)
    })
    df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    # Select top and bottom N if specified
    if top_n and top_n < len(df):
        top_df = df.head(top_n)
        bottom_df = df.tail(top_n)
        plot_df = pd.concat([top_df, bottom_df])
        title = f'Top {top_n} & Bottom {top_n} Students - Accuracy'
    else:
        plot_df = df
        title = 'Per-Student Accuracy'
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['green' if acc >= 0.8 else 'orange' if acc >= 0.6 else 'red' 
              for acc in plot_df['Accuracy']]
    
    bars = ax.barh(range(len(plot_df)), plot_df['Accuracy'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['Student'], fontsize=8)
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add accuracy labels
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(row['Accuracy'] + 0.02, i, f"{row['Accuracy']:.1%} ({row['Correct']}/{row['Total']})", 
                va='center', fontsize=7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='≥80%'),
        Patch(facecolor='orange', alpha=0.7, label='60-80%'),
        Patch(facecolor='red', alpha=0.7, label='<60%')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class accuracy plot saved to: {save_path}")
    
    plt.show()
    
    return df


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot comparison antara multiple models.
    
    Args:
        results: Dictionary dengan format:
                 {'Model1': {'accuracy': 0.9, 'f1': 0.88, ...}, 
                  'Model2': {...}}
        save_path: Path untuk save plot (optional)
        figsize: Figure size
        
    Example:
        >>> results = {
        ...     'CNN (FaceNet)': {'accuracy': 0.92, 'precision': 0.91, 'recall': 0.90, 'f1': 0.90},
        ...     'Transformer (DeiT)': {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.83, 'f1': 0.83}
        ... }
        >>> plot_model_comparison(results)
    """
    # Prepare data
    models = list(results.keys())
    metrics = list(results[models[0]].keys())
    
    data = {metric: [results[model][metric] for model in models] 
            for metric in metrics}
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, model in enumerate(models):
        values = [results[model][metric] for metric in metrics]
        offset = width * (i - len(models) / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=model, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Test training history
    history = {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_loss': [0.9, 0.7, 0.5, 0.4, 0.35],
        'train_acc': [0.7, 0.8, 0.85, 0.9, 0.92],
        'val_acc': [0.65, 0.75, 0.8, 0.85, 0.87]
    }
    plot_training_history(history)
    
    # Test model comparison
    results = {
        'CNN (FaceNet)': {'accuracy': 0.92, 'precision': 0.91, 'recall': 0.90, 'f1': 0.90},
        'Transformer (DeiT)': {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.83, 'f1': 0.83}
    }
    plot_model_comparison(results)
    
    print("✓ All tests completed!")
