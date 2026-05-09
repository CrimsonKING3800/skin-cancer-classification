"""
evaluate.py — Evaluation utilities for skin cancer classification.

Computes comprehensive metrics beyond accuracy: precision, recall, F1,
sensitivity, specificity, and ROC-AUC — critical for medical ML.
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred_proba: Predicted probabilities (from sigmoid output).
        threshold: Decision threshold for binary classification.

    Returns:
        Dictionary of metric names and values.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall (sensitivity)": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }
    return metrics


def print_evaluation_report(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5):
    """Print a formatted evaluation report."""
    metrics = compute_metrics(y_true, y_pred_proba, threshold)

    print("\n" + "=" * 50)
    print("         EVALUATION REPORT")
    print("=" * 50)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:.<30s} {value:.4f}")
        else:
            print(f"  {name:.<30s} {value}")
    print("=" * 50)

    # Full sklearn classification report
    y_pred = (y_pred_proba >= threshold).astype(int)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))
