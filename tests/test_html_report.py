"""Unit tests for standalone interactive HTML reporting."""

from pathlib import Path

from rag_autopsy.reports.html_report import generate_interactive_html_report


def test_generate_interactive_html_report(tmp_path: Path):
    summary = {
        "regime": "regime_a",
        "target_split": "test",
        "validity_summary": {
            "total_attempted": 20,
            "total_valid": 19,
            "overall_validity_rate": 0.95,
        },
        "diagnostic_results": {
            "d4_combined_policy": {
                "metrics": {
                    "macro_f1": 0.72,
                    "balanced_accuracy": 0.75,
                    "coverage": 0.80,
                    "conditional_accuracy": 0.85,
                },
                "bootstrap": {
                    "macro_f1": {"ci_lower": 0.65, "ci_upper": 0.78},
                },
            }
        },
    }

    trials = [
        {
            "trial_id": "trial_001",
            "question_id": "q1",
            "question": "What is HotpotQA?",
            "injected_fault": "F0",
            "trial_valid": True,
            "d4_combined_policy_pred": "F0",
            "answer": "A multi-hop QA benchmark.",
            "generator_context": "HotpotQA is a QA benchmark.",
        }
    ]

    out_file = tmp_path / "test_report.html"
    res_path = generate_interactive_html_report(summary, trials, str(out_file))

    assert Path(res_path).is_file()
    assert out_file.stat().st_size > 1000

    content = out_file.read_text(encoding="utf-8")
    assert "RAG-Autopsy Benchmark Report" in content
    assert "REGIME: REGIME_A" in content
    assert "trial_001" in content
    assert "What is HotpotQA?" in content
