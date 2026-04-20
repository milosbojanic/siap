import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute all classification metrics at a given threshold.

    Returns dict with: auc_roc, accuracy, recall, precision, f1,
    confusion_matrix, classification_report.
    """
    preds = (probs >= threshold).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5

    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = 0.0

    return {
        "auc_roc": auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy_score(labels, preds),
        "recall": recall_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds),
        "classification_report": classification_report(
            labels, preds,
            target_names=["Benign", "Malignant"],
            zero_division=0,
        ),
        "threshold": threshold,
    }


def compute_metrics_at_best_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
) -> Tuple[Dict, float]:
    """
    Find the threshold that maximizes Youden's J statistic (TPR - FPR),
    then compute all metrics at that threshold.

    Returns: (metrics_dict, optimal_threshold)
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    metrics = compute_metrics(labels, probs, threshold=best_threshold)
    return metrics, best_threshold
