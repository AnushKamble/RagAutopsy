"""Tests for core schemas and data contract validation."""

from typing import Any

from rag_autopsy.core.schemas import (
    Diagnosis,
    DiagnosticTrace,
    EvaluatorRecord,
    RetrievedPassage,
)


def test_retrieved_passage_schema() -> None:
    passage = RetrievedPassage(
        passage_id="p001",
        rank=1,
        score=12.5,
        text="Sample text content.",
    )
    assert passage.passage_id == "p001"
    assert passage.rank == 1
    assert passage.score == 12.5


def test_diagnostic_trace_schema(sample_diagnostic_trace: DiagnosticTrace) -> None:
    assert sample_diagnostic_trace.trial_id == "trial_sample_001"
    assert len(sample_diagnostic_trace.retrieved_top_k) == 2
    assert sample_diagnostic_trace.candidate_pool_scores == [9.5, 8.2, 5.0, 3.1, 1.2]
    assert sample_diagnostic_trace.answer == "Vltava River"


def test_evaluator_record_schema(sample_evaluator_record: EvaluatorRecord) -> None:
    assert sample_evaluator_record.trial_id == "trial_sample_001"
    assert sample_evaluator_record.injected_fault == "F0"
    assert sample_evaluator_record.trial_valid is True
    assert sample_evaluator_record.answer_exact_match is True


def test_diagnosis_schema() -> None:
    diag = Diagnosis(
        trial_id="trial_sample_001",
        diagnostic_name="d1_retrieval_rules",
        predicted_cause="F0",
        confidence=0.92,
        abstained=False,
        evidence_rationale="High retrieval score and context support observed.",
        runtime_ms=1.4,
    )
    assert diag.predicted_cause == "F0"
    assert diag.confidence == 0.92
    assert diag.abstained is False


def test_synthetic_qa_fixture_conformance(synthetic_qa_data: list[dict[str, Any]]) -> None:
    assert len(synthetic_qa_data) >= 2
    for item in synthetic_qa_data:
        assert "_id" in item
        assert "question" in item
        assert "answer" in item
        assert "supporting_facts" in item
        assert "context" in item
        assert len(item["context"]) == 10  # 10 paragraphs per distractor setting
        assert len(item["supporting_facts"]) == 2  # exactly 2 supporting facts
