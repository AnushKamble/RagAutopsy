"""Unit tests for the Systematic Ablation Study Suite."""

from pathlib import Path

from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.data.splits import create_question_splits
from rag_autopsy.evaluation.ablations import run_ablation_study_suite


def test_ablation_study_suite_execution(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)

    # Build question split
    manifest = create_question_splits(records, dev_ratio=0.5, calibration_ratio=0.0, test_ratio=0.5, seed=42)

    report = run_ablation_study_suite(
        records=records,
        split_manifest=manifest,
        target_split="dev",
        seed=42,
    )

    # 1. Diagnostic ablation
    assert len(report.diagnostic_ablation) >= 4
    diag_names = [e.diagnostic_name for e in report.diagnostic_ablation]
    assert any("d4" in name for name in diag_names)
    assert any("d1" in name for name in diag_names)

    # 2. Abstention ablation
    assert len(report.abstention_ablation) == 2
    modes = [e.policy_mode for e in report.abstention_ablation]
    assert any("With Predeclared Abstention" in m for m in modes)
    assert any("Forced Prediction" in m for m in modes)

    # 3. Combined faults ablation
    assert len(report.combined_faults_ablation) == 2

    # 4. Severity ablations
    assert len(report.distractor_severity_ablation) > 0
    assert len(report.context_budget_ablation) > 0
    assert "Systematic ablation suite verified" in report.predeclared_findings
