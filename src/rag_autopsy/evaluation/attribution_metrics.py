"""Attribution evaluation metrics: Macro-F1, balanced accuracy, confusion matrices, and coverage."""

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_attribution_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
    abstention_label: str = "UNKNOWN",
) -> dict[str, Any]:
    """Compute comprehensive multi-class attribution metrics with abstention handling.
    
    Args:
        y_true: Ground-truth injected fault labels (e.g. ["F0", "F1", "F2", ...])
        y_pred: Predicted cause labels (can include abstention_label)
        labels: Expected list of true class labels (default: ["F0", "F1", "F2", "F3", "F4"])
        abstention_label: Token indicating diagnostic abstention (default: "UNKNOWN")
    
    Returns:
        Dictionary containing macro_f1, balanced_accuracy, per_class metrics,
        coverage, conditional accuracy, and confusion matrices.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    total_trials = len(y_true)
    if total_trials == 0:
        return {
            "total_trials": 0,
            "coverage": 0.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
        }

    # 1. Coverage / Abstention calculation
    non_abstained_indices = [i for i, pred in enumerate(y_pred) if pred != abstention_label]
    covered_trials = len(non_abstained_indices)
    coverage = round(covered_trials / total_trials, 4)

    # 2. Determine class labels
    if labels is None:
        target_labels = sorted(list(set(y_true)))
    else:
        target_labels = list(labels)

    # 3. Overall Accuracy across all trials
    overall_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    overall_accuracy = round(overall_correct / total_trials, 4)

    # 4. Metrics on covered (non-abstained) subset
    if covered_trials > 0:
        y_true_covered = [y_true[i] for i in non_abstained_indices]
        y_pred_covered = [y_pred[i] for i in non_abstained_indices]
        covered_correct = sum(1 for yt, yp in zip(y_true_covered, y_pred_covered) if yt == yp)
        conditional_accuracy = round(covered_correct / covered_trials, 4)
    else:
        conditional_accuracy = 0.0
        y_true_covered = []
        y_pred_covered = []

    # 5. Confusion matrix over target classes
    # If a prediction is UNKNOWN, it acts as an error across classes unless mapped
    # For strict attribution, evaluate across target_labels
    y_pred_mapped = [yp if yp in target_labels else "UNMAPPED_OR_UNKNOWN" for yp in y_pred]
    eval_labels = target_labels + ["UNMAPPED_OR_UNKNOWN"]

    raw_cm = confusion_matrix(y_true, y_pred_mapped, labels=eval_labels)

    # True-cause normalized confusion matrix (rows sum to 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = raw_cm.sum(axis=1, keepdims=True)
        norm_cm = np.nan_to_num(raw_cm.astype(float) / row_sums)

    # 6. Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred_mapped,
        labels=target_labels,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float]] = {}
    for idx, label in enumerate(target_labels):
        per_class[label] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    macro_f1 = round(float(np.mean(f1)), 4)
    balanced_accuracy = round(float(np.mean(recall)), 4)

    return {
        "total_trials": total_trials,
        "covered_trials": covered_trials,
        "coverage": coverage,
        "overall_accuracy": overall_accuracy,
        "conditional_accuracy": conditional_accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "target_labels": target_labels,
        "per_class": per_class,
        "raw_confusion_matrix": raw_cm.tolist(),
        "normalized_confusion_matrix": np.round(norm_cm, 4).tolist(),
        "confusion_matrix_labels": eval_labels,
    }
