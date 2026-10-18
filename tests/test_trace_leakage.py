"""Tests ensuring strict structural isolation between visible traces and privileged evaluator data."""

import pytest
from pydantic import ValidationError
from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord, RetrievedPassage


def test_diagnostic_trace_has_no_evaluator_fields() -> None:
    """Verify that DiagnosticTrace model schema does not define privileged fields."""
    forbidden_fields = {
        "injected_fault",
        "fault_parameters",
        "gold_answer",
        "gold_answers",
        "gold_evidence_ids",
        "target_rank",
        "injection_applied",
        "trial_valid",
        "invalid_reason",
    }
    model_fields = set(DiagnosticTrace.model_fields.keys())
    intersection = model_fields.intersection(forbidden_fields)
    assert not intersection, f"DiagnosticTrace schema leaks privileged fields: {intersection}"


def test_leakage_rejection_in_metadata() -> None:
    """Ensure that attempts to smuggle privileged fields inside metadata dict raise a ValueError."""
    passages = [
        RetrievedPassage(passage_id="p1", rank=1, score=5.0, text="Sample text")
    ]
    with pytest.raises(ValidationError, match="Leakage violation"):
        DiagnosticTrace(
            trial_id="opaque_trial_999",
            question_id="q001",
            question="What is X?",
            retrieved_top_k=passages,
            pre_assembly_passage_ids=["p1"],
            post_assembly_passage_ids=["p1"],
            generator_context="Sample text",
            answer="X is 1",
            latency_ms=10.0,
            metadata={"injected_fault": "F1"},  # Illegal smuggling!
        )


def test_leakage_rejection_in_trial_id() -> None:
    """Ensure that encoding the fault type in trial_id raises a ValueError."""
    passages = [
        RetrievedPassage(passage_id="p1", rank=1, score=5.0, text="Sample text")
    ]
    with pytest.raises(ValidationError, match="Leakage violation"):
        DiagnosticTrace(
            trial_id="trial-f1-severe",  # Illegal encoding!
            question_id="q001",
            question="What is X?",
            retrieved_top_k=passages,
            pre_assembly_passage_ids=["p1"],
            post_assembly_passage_ids=["p1"],
            generator_context="Sample text",
            answer="X is 1",
            latency_ms=10.0,
        )


def test_evaluator_record_contains_required_evaluator_fields(
    sample_evaluator_record: EvaluatorRecord,
) -> None:
    """Verify that EvaluatorRecord contains the necessary ground truth fields."""
    assert hasattr(sample_evaluator_record, "injected_fault")
    assert hasattr(sample_evaluator_record, "gold_answers")
    assert hasattr(sample_evaluator_record, "gold_evidence_ids")
    assert hasattr(sample_evaluator_record, "trial_valid")
