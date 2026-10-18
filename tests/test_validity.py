"""Unit tests for injection validity adjudication."""

from rag_autopsy.core.schemas import EvaluatorRecord
from rag_autopsy.evaluation.validity import adjudicate_trials


def test_adjudicate_trials() -> None:
    records = [
        EvaluatorRecord(
            trial_id="t1",
            question_id="q1",
            split="dev",
            injected_fault="F1",
            gold_answers=["A1"],
            gold_evidence_ids=["p1"],
            trial_valid=True,
        ),
        EvaluatorRecord(
            trial_id="t2",
            question_id="q1",
            split="dev",
            injected_fault="F1",
            gold_answers=["A1"],
            gold_evidence_ids=["p1"],
            trial_valid=False,
            invalid_reason="Precondition failed: gold evidence was not in corpus.",
        ),
        EvaluatorRecord(
            trial_id="t3",
            question_id="q2",
            split="dev",
            injected_fault="F2",
            gold_answers=["A2"],
            gold_evidence_ids=["p2"],
            trial_valid=True,
        ),
    ]

    valid, invalid, summary = adjudicate_trials(records)
    assert len(valid) == 2
    assert len(invalid) == 1
    assert summary.total_attempted == 3
    assert summary.total_valid == 2
    assert summary.total_invalid == 1
    assert summary.overall_validity_rate == 0.6667

    f1_stats = summary.by_fault["F1"]
    assert f1_stats.attempted == 2
    assert f1_stats.valid == 1
    assert f1_stats.invalid == 1
    assert "Precondition failed" in list(f1_stats.invalid_reasons.keys())[0]
