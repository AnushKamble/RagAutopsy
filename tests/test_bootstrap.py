"""Unit tests for clustered bootstrap uncertainty estimation."""

import pytest

from rag_autopsy.evaluation.bootstrap import clustered_bootstrap_attribution


def test_clustered_bootstrap_bounds() -> None:
    # 5 questions, each with 2 trials
    question_ids = ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4", "q5", "q5"]
    y_true = ["F0", "F1", "F0", "F1", "F0", "F1", "F0", "F1", "F0", "F1"]
    y_pred = ["F0", "F1", "F0", "F1", "F0", "F0", "F0", "F1", "F0", "F1"]  # 1 error on q3

    summary = clustered_bootstrap_attribution(
        question_ids=question_ids,
        y_true=y_true,
        y_pred=y_pred,
        labels=["F0", "F1"],
        n_bootstraps=100,
        alpha=0.05,
        seed=42,
    )

    assert summary.n_bootstraps == 100
    assert summary.macro_f1.ci_lower <= summary.macro_f1.mean <= summary.macro_f1.ci_upper
    assert summary.balanced_accuracy.ci_lower <= summary.balanced_accuracy.mean <= summary.balanced_accuracy.ci_upper
    assert summary.overall_accuracy.ci_lower <= summary.overall_accuracy.mean <= summary.overall_accuracy.ci_upper


def test_clustered_bootstrap_length_mismatch() -> None:
    with pytest.raises(ValueError):
        clustered_bootstrap_attribution(
            question_ids=["q1"],
            y_true=["F0", "F1"],
            y_pred=["F0"],
        )
