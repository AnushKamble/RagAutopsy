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


