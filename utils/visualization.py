"""
Visualization utilities for training curves, ROC curves,
confusion matrices, and PR curves (reproducing paper Figs. 8-15).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


CLASS_NAMES = ["CN", "MCI", "AD"]
PALETTE = {
    "CN":  "#1f77b4",
    "MCI": "#ff7f0e",
    "AD":  "#d62728",
}


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Training curves (Fig. 14)
# ---------------------------------------------------------------------------

def plot_training_curves(history: Dict[str, list], output_dir: str = "results/figures") -> None:
    """Plot accuracy and loss curves over epochs."""
    d = _ensure_dir(output_dir)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy
    ax1.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#1f77b4")
    ax1.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val Accuracy",   color="#ff7f0e")
    if "test_acc" in history:
        best_test = max(history["test_acc"]) * 100
        ax1.axhline(best_test, linestyle="--", color="#d62728", alpha=0.7, label=f"Test Acc ({best_test:.1f}%)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)"); ax1.set_title("Accuracy Curves")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4")
    ax2.plot(epochs, history["val_loss"],   label="Val Loss",   color="#ff7f0e")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("Loss Curves")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(d / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {d / 'training_curves.png'}")


# ---------------------------------------------------------------------------
# Confusion matrix (Fig. 12)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    output_path: str = "results/figures/confusion_matrix.png",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, cbar_kws={"label": "Normalised count"},
    )
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    _ensure_dir(str(Path(output_path).parent))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# ROC curves (Fig. 10)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    model_name: str = "MDLF",
    output_dir: str = "results/figures",
) -> None:
    d = _ensure_dir(output_dir)
    n_classes = y_probs.shape[1]
    y_bin = np.eye(n_classes)[y_true]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Per-class (left)
    for k, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, k], y_probs[:, k])
        auc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=list(PALETTE.values())[k],
                 label=f"{cls} (AUC={auc_val:.3f})", lw=2)
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"Per-Class ROC — {model_name}")
    ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

    # Macro-average (right) — single curve
    fpr_grid = np.linspace(0, 1, 200)
    tprs = []
    for k in range(n_classes):
        fpr_k, tpr_k, _ = roc_curve(y_bin[:, k], y_probs[:, k])
        tprs.append(np.interp(fpr_grid, fpr_k, tpr_k))
    mean_tpr = np.mean(tprs, axis=0)
    macro_auc = auc(fpr_grid, mean_tpr)
    ax2.plot(fpr_grid, mean_tpr, color="#1f77b4", lw=2.5,
             label=f"{model_name} (AUC={macro_auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Macro-Average ROC Comparison")
    ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(d / f"roc_curves_{model_name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Precision-Recall curves (Fig. 11)
# ---------------------------------------------------------------------------

def plot_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    model_name: str = "MDLF",
    output_dir: str = "results/figures",
) -> None:
    d = _ensure_dir(output_dir)
    n_classes = y_probs.shape[1]
    y_bin = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(7, 5))
    for k, cls in enumerate(CLASS_NAMES):
        prec, rec, _ = precision_recall_curve(y_bin[:, k], y_probs[:, k])
        ap = average_precision_score(y_bin[:, k], y_probs[:, k])
        ax.plot(rec, prec, color=list(PALETTE.values())[k],
                label=f"{cls} (AP={ap:.3f})", lw=2)

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Per-Class Precision-Recall — {model_name}")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(d / f"pr_curves_{model_name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Bar chart comparison (Fig. 8)
# ---------------------------------------------------------------------------

def plot_model_comparison(
    model_results: Dict[str, Dict],
    output_dir: str = "results/figures",
) -> None:
    """
    Bar chart comparing Accuracy, Precision, Recall, F1 across models.
    Reproduces Fig. 8 from the paper.
    """
    d = _ensure_dir(output_dir)
    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    model_names = list(model_results.keys())
    x = np.arange(len(model_names))
    width = 0.20

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.5), 6))
    for i, (key, label, col) in enumerate(zip(metric_keys, metric_labels, colors)):
        vals = [model_results[m].get(key, 0) * 100 for m in model_names]
        ax.bar(x + i * width, vals, width - 0.02, label=label, color=col, alpha=0.85)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)"); ax.set_title("Model Performance Comparison")
    ax.legend(); ax.set_ylim(60, 100); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(d / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison chart saved to {d / 'model_comparison.png'}")
