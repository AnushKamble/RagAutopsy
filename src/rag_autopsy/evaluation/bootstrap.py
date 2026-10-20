"""Clustered bootstrap uncertainty estimation at the Question ID level."""

import random
from collections import defaultdict

import numpy as np
from pydantic import BaseModel

from rag_autopsy.evaluation.attribution_metrics import compute_attribution_metrics


class BootstrapInterval(BaseModel):
    mean: float
    ci_lower: float
    ci_upper: float


class AttributionBootstrapSummary(BaseModel):
    n_bootstraps: int
    alpha: float
    macro_f1: BootstrapInterval
    balanced_accuracy: BootstrapInterval
    overall_accuracy: BootstrapInterval


def clustered_bootstrap_attribution(
    question_ids: list[str],
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> AttributionBootstrapSummary:
    """Perform non-parametric clustered bootstrap resampling grouped by Question ID.
    
    Guarantees that all fault variants of a question are kept clustered together
    during resampling to prevent variance underestimation.
    """
    if len(question_ids) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Inputs question_ids, y_true, and y_pred must have identical length.")

    # Group trial indices by question_id
    q_to_indices = defaultdict(list)
    for idx, qid in enumerate(question_ids):
        q_to_indices[qid].append(idx)

    unique_qids = sorted(list(q_to_indices.keys()))
    n_questions = len(unique_qids)

    if n_questions == 0:
        empty = BootstrapInterval(mean=0.0, ci_lower=0.0, ci_upper=0.0)
        return AttributionBootstrapSummary(
            n_bootstraps=0, alpha=alpha, macro_f1=empty, balanced_accuracy=empty, overall_accuracy=empty
        )

    rng = random.Random(seed)

    boot_macro_f1: list[float] = []
    boot_bal_acc: list[float] = []
    boot_acc: list[float] = []

    for _ in range(n_bootstraps):
        sampled_qids = rng.choices(unique_qids, k=n_questions)
        sampled_indices: list[int] = []
        for qid in sampled_qids:
            sampled_indices.extend(q_to_indices[qid])

        sub_true = [y_true[i] for i in sampled_indices]
        sub_pred = [y_pred[i] for i in sampled_indices]

        metrics = compute_attribution_metrics(sub_true, sub_pred, labels=labels)
        boot_macro_f1.append(metrics["macro_f1"])
        boot_bal_acc.append(metrics["balanced_accuracy"])
        boot_acc.append(metrics["overall_accuracy"])

    lower_p = (alpha / 2.0) * 100.0
    upper_p = (1.0 - alpha / 2.0) * 100.0

    return AttributionBootstrapSummary(
        n_bootstraps=n_bootstraps,
        alpha=alpha,
        macro_f1=BootstrapInterval(
            mean=round(float(np.mean(boot_macro_f1)), 4),
            ci_lower=round(float(np.percentile(boot_macro_f1, lower_p)), 4),
            ci_upper=round(float(np.percentile(boot_macro_f1, upper_p)), 4),
        ),
        balanced_accuracy=BootstrapInterval(
            mean=round(float(np.mean(boot_bal_acc)), 4),
            ci_lower=round(float(np.percentile(boot_bal_acc, lower_p)), 4),
            ci_upper=round(float(np.percentile(boot_bal_acc, upper_p)), 4),
        ),
        overall_accuracy=BootstrapInterval(
            mean=round(float(np.mean(boot_acc)), 4),
            ci_lower=round(float(np.percentile(boot_acc, lower_p)), 4),
            ci_upper=round(float(np.percentile(boot_acc, upper_p)), 4),
        ),
    )
