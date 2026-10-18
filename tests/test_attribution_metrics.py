"""Unit tests for attribution metrics and confusion matrix calculations."""

import pytest
from rag_autopsy.evaluation.attribution_metrics import compute_attribution_metrics


def test_perfect_attribution() -> None:
    y_true = ["F0", "F1", "F2", "F3", "F4"]
    y_pred = ["F0", "F1", "F2", "F3", "F4"]

    metrics = compute_attribution_metrics(y_true, y_pred, labels=["F0", "F1", "F2", "F3", "F4"])
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["conditional_accuracy"] == 1.0
    assert metrics["coverage"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_attribution_with_abstention() -> None:
    y_true = ["F0", "F1", "F2", "F3"]
    # F2 is abstained (UNKNOWN), F3 is wrong (predicted F0)
    y_pred = ["F0", "F1", "UNKNOWN", "F0"]

    metrics = compute_attribution_metrics(y_true, y_pred, labels=["F0", "F1", "F2", "F3"])
    assert metrics["total_trials"] == 4
    assert metrics["covered_trials"] == 3
    assert metrics["coverage"] == 0.75

    # Among covered cases: 2 out of 3 are correct (F0, F1)
    assert round(metrics["conditional_accuracy"], 2) == 0.67

    # Overall accuracy: 2 out of 4 correct
    assert metrics["overall_accuracy"] == 0.50


def test_normalized_confusion_matrix_row_sums() -> None:
    y_true = ["F0", "F0", "F1", "F1", "F2"]
    y_pred = ["F0", "F1", "F1", "F1", "F0"]

    metrics = compute_attribution_metrics(y_true, y_pred, labels=["F0", "F1", "F2"])
    norm_cm = metrics["normalized_confusion_matrix"]

    # Each row of normalized CM corresponding to classes with support must sum to 1.0
    for row in norm_cm[:3]:
        assert round(sum(row), 2) == 1.0
