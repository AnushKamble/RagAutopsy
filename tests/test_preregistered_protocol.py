"""Tests verifying the preregistered evaluation protocol invariants, leakage isolation, and identifiability."""

import json
from pathlib import Path
import pytest
import yaml

from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord, RetrievedPassage
from rag_autopsy.data.splits import load_split_manifest
from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic
from rag_autopsy.diagnostics.d2_support import D2LexicalSupportDiagnostic
from rag_autopsy.diagnostics.d4_policy import D4CombinedAttributionPolicy
from rag_autopsy.evaluation.attribution_metrics import compute_attribution_metrics


def test_frozen_protocol_yaml_exists_and_parses() -> None:
    protocol_path = Path("configs/frozen_protocol_v1.yaml")
    assert protocol_path.is_file(), "configs/frozen_protocol_v1.yaml must exist"
    with open(protocol_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["status"] == "FROZEN"
    assert data["protocol_version"] == "1.0.0"
    assert "research_questions" in data
    assert "frozen_thresholds" in data


def test_split_manifest_disjointness_and_grouping() -> None:
    manifest_path = Path("configs/split_manifest_100.json")
    assert manifest_path.is_file(), "configs/split_manifest_100.json must exist"
    manifest = load_split_manifest(manifest_path)

    dev_set = set(manifest.dev_ids)
    cal_set = set(manifest.calibration_ids)
    test_set = set(manifest.test_ids)

    # Disjointness invariants
    assert dev_set.intersection(cal_set) == set(), "dev and calibration must not overlap"
    assert dev_set.intersection(test_set) == set(), "dev and test must not overlap"
    assert cal_set.intersection(test_set) == set(), "calibration and test must not overlap"

    # Exactly 100 questions partitioned 60/20/20
    assert len(dev_set) == 60
    assert len(cal_set) == 20
    assert len(test_set) == 20
    assert len(dev_set | cal_set | test_set) == 100


def test_trace_schema_strict_leakage_absence() -> None:
    """Ensure DiagnosticTrace JSON dump never contains any forbidden evaluator fields."""
    forbidden = [
        "injected_fault",
        "fault_parameters",
        "gold_answer",
        "gold_answers",
        "gold_evidence_ids",
        "target_rank",
        "injection_applied",
        "trace_condition_achieved",
        "trial_valid",
        "invalid_reason",
    ]

    trace = DiagnosticTrace(
        trial_id="trial_leakage_check",
        question_id="q001",
        question="What is this?",
        retrieved_top_k=[
            RetrievedPassage(passage_id="p1", rank=1, score=5.0, text="Context text."),
        ],
        pre_assembly_passage_ids=["p1"],
        post_assembly_passage_ids=["p1"],
        generator_context="Context text.",
        answer="Context text.",
        latency_ms=1.0,
    )
    dump_dict = trace.model_dump()
    for f in forbidden:
        assert f not in dump_dict, f"Forbidden field '{f}' present in DiagnosticTrace model dump!"


def test_identifiability_regime_b_abstention() -> None:
    """Verify that in Regime B (blind top-k), D1 and D4 abstain on retrieval miss rather than faking diagnosis."""
    # Under regime B with a low top score, F1 vs F2 is unidentifiable
    trace_b = DiagnosticTrace(
        trial_id="trial_regime_b_unidentifiable",
        question_id="q002",
        question="Query?",
        retrieved_top_k=[
            RetrievedPassage(passage_id="p_distractor", rank=1, score=0.8, text="Irrelevant distractor."),
        ],
        candidate_pool_scores=None,  # Blind top-k: candidate pool not available!
        pre_assembly_passage_ids=["p_distractor"],
        post_assembly_passage_ids=["p_distractor"],
        generator_context="Irrelevant distractor.",
        answer="Unfounded answer",
        latency_ms=2.0,
    )

    d1_regime_b = D1RetrievalDiagnostic(regime="regime_b", low_score_threshold=2.0, abstain_on_ambiguity=True)
    res_d1 = d1_regime_b.diagnose(trace_b)
    assert res_d1.abstained is True
    assert "AMBIGUOUS_RETRIEVAL_MISS" in res_d1.predicted_cause or res_d1.predicted_cause == "UNKNOWN"

    d4_regime_b = D4CombinedAttributionPolicy(regime="regime_b", low_score_threshold=2.0, abstention_enabled=True)
    res_d4 = d4_regime_b.diagnose(trace_b)
    assert res_d4.abstained is True


def test_report_metric_computation_properties() -> None:
    """Validate mathematical sanity of attribution metrics computation."""
    y_true = ["F0", "F1", "F2", "F3", "F4", "F0"]
    y_pred = ["F0", "F1", "F2", "F3", "UNKNOWN", "F1"]

    metrics = compute_attribution_metrics(y_true, y_pred, labels=["F0", "F1", "F2", "F3", "F4"])

    assert metrics["total_trials"] == 6
    assert metrics["covered_trials"] == 5
    assert round(metrics["coverage"], 4) == round(5 / 6, 4)

    # 4 correct out of 5 covered trials (F0, F1, F2, F3)
    assert metrics["conditional_accuracy"] == 0.8
    # 4 correct out of 6 total trials
    assert round(metrics["overall_accuracy"], 4) == round(4 / 6, 4)

    # Normalized confusion matrix rows corresponding to supported classes must sum to 1.0
    norm_cm = metrics["normalized_confusion_matrix"]
    for row in norm_cm[:5]:
        assert round(sum(row), 2) == 1.0
