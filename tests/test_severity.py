"""Unit tests for fault severity scaling and sensitivity sweeps."""

from pathlib import Path

from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.evaluation.severity import (
    run_severity_analysis,
    sweep_context_budget_severity,
    sweep_distractor_severity,
)
from rag_autopsy.pipeline.corpus import Corpus


def test_severity_sweeps(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)
    corpus = Corpus.from_qa_records(records)

    # 1. Distractor sweep (D in [1, 3, 5])
    d_curve = sweep_distractor_severity(
        records=records,
        base_corpus=corpus,
        distractor_counts=[1, 3, 5],
        cutoff_k=3,
        seed=42,
    )
    assert len(d_curve) == 3
    for pt in d_curve:
        assert 0.0 <= pt.validity_rate <= 1.0
        assert 0.0 <= pt.d1_f2_recall <= 1.0
        assert 0.0 <= pt.d4_f2_recall <= 1.0
        assert pt.trials_attempted == len(records)

    # 2. Context budget sweep (C in [500, 2000])
    c_curve = sweep_context_budget_severity(
        records=records,
        base_corpus=corpus,
        max_chars_list=[500, 2000],
        top_k=5,
        seed=42,
    )
    assert len(c_curve) == 2
    for pt in c_curve:
        assert 0.0 <= pt.validity_rate <= 1.0
        assert 0.0 <= pt.d2_f3_recall <= 1.0
        assert pt.trials_attempted == len(records)

    # 3. Combined report
    rep = run_severity_analysis(
        records=records,
        base_corpus=corpus,
        distractor_counts=[1, 3],
        context_budgets=[500, 1000],
        cutoff_k=3,
        seed=42,
    )
    assert len(rep.distractor_curve) == 2
    assert len(rep.context_budget_curve) == 2
    assert "fault severity sensitivity" in rep.summary_findings
