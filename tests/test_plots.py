"""Tests for static plotting and evaluation visualization."""

import json
from pathlib import Path

import pytest

from rag_autopsy.reports.plots import (
    generate_all_evaluation_plots,
    plot_bootstrap_macro_f1,
    plot_confusion_matrix,
    plot_regime_comparison,
)


def test_plot_confusion_matrix(tmp_path: Path):
    matrix = [
        [0.8, 0.2, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9],
    ]
    labels = ["F0", "F1", "F2"]
    out_file = tmp_path / "test_cm.png"
    
    fig = plot_confusion_matrix(matrix, labels, title="Unit Test CM", output_path=out_file)
    assert fig is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_plot_regime_comparison(tmp_path: Path):
    matrix_a = [[1.0, 0.0], [0.0, 1.0]]
    matrix_b = [[0.5, 0.5], [0.5, 0.5]]
    labels = ["A", "B"]
    out_file = tmp_path / "test_regime_comp.png"
    
    fig = plot_regime_comparison(matrix_a, matrix_b, labels, output_path=out_file)
    assert fig is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_plot_bootstrap_macro_f1(tmp_path: Path):
    summary_data = {
        "diagnostic_results": {
            "baseline_majority_cause": {
                "bootstrap": {"macro_f1": {"mean": 0.1, "ci_lower": 0.08, "ci_upper": 0.12}},
                "metrics": {"macro_f1": 0.1},
            },
            "d4_combined_policy": {
                "bootstrap": {"macro_f1": {"mean": 0.75, "ci_lower": 0.70, "ci_upper": 0.80}},
                "metrics": {"macro_f1": 0.75},
            },
        }
    }
    out_file = tmp_path / "test_bootstrap.png"
    
    fig = plot_bootstrap_macro_f1(summary_data, output_path=out_file)
    assert fig is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_generate_all_evaluation_plots_e2e(tmp_path: Path):
    # Create mock summaries
    data_a = {
        "diagnostic_results": {
            "d4_combined_policy": {
                "metrics": {
                    "normalized_confusion_matrix": [[1.0, 0.0], [0.0, 1.0]],
                    "confusion_matrix_labels": ["F0", "F1"],
                    "macro_f1": 1.0,
                },
                "bootstrap": {"macro_f1": {"mean": 1.0, "ci_lower": 0.9, "ci_upper": 1.0}},
            }
        }
    }
    data_b = {
        "diagnostic_results": {
            "d4_combined_policy": {
                "metrics": {
                    "normalized_confusion_matrix": [[0.5, 0.5], [0.5, 0.5]],
                    "confusion_matrix_labels": ["F0", "F1"],
                    "macro_f1": 0.5,
                },
                "bootstrap": {"macro_f1": {"mean": 0.5, "ci_lower": 0.4, "ci_upper": 0.6}},
            }
        }
    }
    file_a = tmp_path / "sum_a.json"
    file_b = tmp_path / "sum_b.json"
    with open(file_a, "w", encoding="utf-8") as f:
        json.dump(data_a, f)
    with open(file_b, "w", encoding="utf-8") as f:
        json.dump(data_b, f)
        
    out_dir = tmp_path / "plots"
    generated = generate_all_evaluation_plots(file_a, file_b, out_dir)
    assert "regime_a_d4_cm" in generated
    assert "regime_a_bootstrap_bar" in generated
    assert "regime_comparison_d4" in generated
    for p in generated.values():
        assert Path(p).exists()
        assert Path(p).stat().st_size > 1000


def test_generate_plots_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        generate_all_evaluation_plots("non_existent_file_path.json")
