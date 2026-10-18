"""Tests for the RAG-Autopsy CLI."""

from typer.testing import CliRunner
from rag_autopsy.cli import app
from rag_autopsy import __version__

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "RAG-Autopsy" in result.stdout
    assert "smoke" in result.stdout
    assert "version" in result.stdout
    assert "info" in result.stdout


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_info() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Python Version" in result.stdout
    assert "BM25" in result.stdout


def test_cli_smoke() -> None:
    result = runner.invoke(app, ["smoke"])
    assert result.exit_code == 0
    assert "Smoke test PASSED successfully" in result.stdout


def test_cli_evaluate(tmp_path) -> None:
    result = runner.invoke(app, ["evaluate", "--split", "test", "--bootstraps", "10", "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert "Diagnostic Attribution Benchmark" in result.stdout
    assert (tmp_path / "evaluation_summary.json").is_file()
    assert (tmp_path / "trial_predictions.csv").is_file()


def test_cli_evaluate_with_f5_and_d3(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["evaluate", "--split", "test", "--bootstraps", "5", "--output", str(tmp_path), "--include-f5", "--include-d3"],
    )
    assert result.exit_code == 0
    assert "Diagnostic Attribution Benchmark" in result.stdout
    assert (tmp_path / "evaluation_summary.json").is_file()


def test_cli_generate_plots(tmp_path) -> None:
    result = runner.invoke(app, ["generate-plots", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Successfully generated" in result.stdout
    assert (tmp_path / "regime_a_d4_confusion_matrix.png").is_file()
    assert (tmp_path / "regime_a_macro_f1_bootstrap.png").is_file()


def test_cli_demo() -> None:
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "RAG-Autopsy: Fault-Injection-Validated Failure Diagnosis" in result.stdout
    assert "Step 1: Clean Pipeline Control" in result.stdout
    assert "Step 2: Fault Injection F3" in result.stdout
    assert "Step 3: Identifiability Demonstration" in result.stdout
    assert "Demonstration completed successfully" in result.stdout


def test_cli_report(tmp_path) -> None:
    # First evaluate to generate summary
    runner.invoke(app, ["evaluate", "--split", "test", "--bootstraps", "5", "--output", str(tmp_path)])
    out_html = tmp_path / "report.html"
    result = runner.invoke(
        app,
        [
            "report",
            "--summary",
            str(tmp_path / "evaluation_summary.json"),
            "--predictions",
            str(tmp_path / "trial_predictions.csv"),
            "--output",
            str(out_html),
        ],
    )
    assert result.exit_code == 0
    assert "Successfully generated interactive HTML report" in result.stdout
    assert out_html.is_file()


def test_cli_ablate(tmp_path) -> None:
    result = runner.invoke(app, ["ablate", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Systematic Ablation Suite" in result.stdout
    assert "Ablation 1: Diagnostic Families" in result.stdout
    assert (tmp_path / "ablation_study_report.json").is_file()


def test_cli_severity(tmp_path) -> None:
    result = runner.invoke(app, ["severity", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Fault Severity Sensitivity Analysis" in result.stdout
    assert (tmp_path / "severity_report.json").is_file()


def test_cli_ragtruth(tmp_path) -> None:
    result = runner.invoke(app, ["ragtruth", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "RAGTruth Grounding & Hallucination Benchmark" in result.stdout
    assert (tmp_path / "ragtruth_benchmark_report.json").is_file()


def test_cli_quality_compare(tmp_path) -> None:
    result = runner.invoke(app, ["quality-compare", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Comparing Scalar RAG Quality Metrics vs Fault Attribution" in result.stdout
    assert (tmp_path / "quality_vs_attribution.json").is_file()



