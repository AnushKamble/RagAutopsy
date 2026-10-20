"""Unit and integration tests for the natural failure audit, 8-dimensional rubric, and transfer analysis."""

from pathlib import Path

import pytest

from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.evaluation.natural_audit import (
    export_natural_audit_report,
    run_natural_failure_audit,
)


@pytest.fixture
def sample_records():
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "synthetic_qa.json"
    raw = load_raw_hotpot_json(fixture_path)
    return parse_hotpot_records(raw)


def test_natural_audit_execution_and_rubric_dimensions(sample_records) -> None:
    """Verify natural audit runs clean pipeline and populates all 8 rubric dimensions."""
    audit_data = run_natural_failure_audit(sample_records, regime="regime_a")

    assert "total_evaluated_clean_questions" in audit_data
    assert "total_natural_failures" in audit_data
    assert "reviewer_agreement" in audit_data
    assert "transfer_analysis" in audit_data
    assert "failures" in audit_data

    failures = audit_data["failures"]
    assert len(failures) == audit_data["total_natural_failures"]

    for f in failures:
        # Rubric dimensions 1-5
        assert f["answer_correctness"] in ["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]
        assert f["support_in_context"] in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
        assert f["evidence_retrieval"] in ["RETRIEVED_FULL", "RETRIEVED_PARTIAL", "MISSED"]
        assert f["context_survival"] in ["SURVIVED", "TRUNCATED", "NOT_APPLICABLE"]
        assert f["context_contradiction"] in ["CONTRADICTION_PRESENT", "CONSISTENT"]

        # Rubric dimensions 6-8
        assert f["likely_primary_cause"] in [
            "REASONING_ERROR",
            "F2_RANKED_OUT",
            "F1_UNAVAILABLE",
            "F3_ASSEMBLY_LOSS",
            "F4_CONTRADICTION",
        ]
        assert isinstance(f["multiple_plausible_causes"], list)
        assert isinstance(f["is_unknown"], bool)
        assert len(f["reviewer_rationale"]) > 0

        # Reviewer tracking
        assert f["reviewer_1_cause"] != ""
        assert f["reviewer_2_cause"] != ""
        assert f["adjudicated_cause"] != ""


def test_natural_audit_reviewer_agreement_properties(sample_records) -> None:
    """Verify that inter-annotator agreement metrics are mathematically bounded."""
    audit_data = run_natural_failure_audit(sample_records, regime="regime_a")
    agr = audit_data["reviewer_agreement"]

    assert 0.0 <= agr["percent_agreement"] <= 1.0
    assert -1.0 <= agr["cohen_kappa"] <= 1.0
    assert agr["total_disagreements"] >= 0
    assert 0.0 <= agr["disagreement_rate"] <= 1.0
    assert "methodology_note" in agr


def test_natural_audit_transfer_analysis(sample_records) -> None:
    """Verify transfer analysis evaluates diagnostic agreement on natural failures."""
    audit_data = run_natural_failure_audit(sample_records, regime="regime_a")
    transfer = audit_data["transfer_analysis"]

    assert "d4_transfer_agreement_rate" in transfer
    assert 0.0 <= transfer["d4_transfer_agreement_rate"] <= 1.0
    assert "breakdown_by_natural_cause" in transfer
    assert "scientific_interpretation" in transfer


def test_natural_audit_export_artifacts(tmp_path, sample_records) -> None:
    """Verify that export_natural_audit_report creates all machine-readable files."""
    audit_data = run_natural_failure_audit(sample_records, regime="regime_a")
    artifacts = export_natural_audit_report(audit_data, output_dir=str(tmp_path))

    assert (tmp_path / "natural_audit_detailed.json").is_file()
    assert (tmp_path / "natural_failure_records.jsonl").is_file()
    assert (tmp_path / "disagreement_summary.json").is_file()
    assert (tmp_path / "transfer_analysis.json").is_file()
