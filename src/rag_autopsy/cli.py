"""Command-line interface for RAG-Autopsy."""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_autopsy import __version__
from rag_autopsy.core.config import ExperimentConfig, load_config
from rag_autopsy.core.logging import setup_logging
from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord, RetrievedPassage
from rag_autopsy.data.adapter import load_qa_records, load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.data.splits import create_question_splits
from rag_autopsy.evaluation.runner import run_experiment_harness
from rag_autopsy.reports.summary import export_evaluation_report

app = typer.Typer(
    name="rag-autopsy",
    help="RAG-Autopsy: Fault-Injection-Validated Failure Attribution for RAG Systems.",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the current version of RAG-Autopsy."""
    console.print(f"[bold cyan]RAG-Autopsy[/bold cyan] version [green]{__version__}[/green]")


@app.command()
def info() -> None:
    """Display environment, compute, and configuration defaults."""
    import platform
    import sys

    table = Table(title="RAG-Autopsy Environment & System Status")
    table.add_column("Property", style="bold yellow")
    table.add_column("Value", style="cyan")

    table.add_row("RAG-Autopsy Version", __version__)
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("Platform", platform.platform())
    table.add_row("Executable", sys.executable)
    table.add_row("Default Dataset", "HotpotQA (Distractor Dev v1.1)")
    table.add_row("Default Retriever", "BM25 (rank-bm25, pure Python/CPU)")
    table.add_row("Default Diagnostics", "D1 (Retrieval Heuristics), D2 (Lexical Support), D4 (Combined)")

    console.print(table)


@app.command()
def smoke(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to custom experiment YAML configuration."
    )
) -> None:
    """Run an end-to-end smoke verification on synthetic test data."""
    setup_logging(level="INFO")
    console.print(Panel("[bold green]Starting RAG-Autopsy Foundation Smoke Test[/bold green]"))

    cfg = load_config(config_path)
    console.print(f"Loaded config: [bold]{cfg.experiment_name}[/bold] (seed: {cfg.random_seed})")

    fixture_path = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "synthetic_qa.json"
    if not fixture_path.is_file():
        console.print(f"[bold red]Synthetic fixture not found at: {fixture_path}[/bold red]")
        raise typer.Exit(code=1)

    raw_items = load_raw_hotpot_json(fixture_path)
    records = parse_hotpot_records(raw_items)
    console.print(f"Loaded and validated [cyan]{len(records)}[/cyan] synthetic QA records.")

    # Generate splits
    manifest = create_question_splits(records, dev_ratio=0.5, calibration_ratio=0.0, test_ratio=0.5, seed=42)
    console.print(f"Generated split manifest with [green]{manifest.total_questions}[/green] questions.")

    # Run quick experiment harness on dev partition
    harness_results = run_experiment_harness(
        qa_records=records,
        split_manifest=manifest,
        target_split="dev",
        n_bootstraps=10,
    )

    validity = harness_results["validity_summary"]
    console.print(
        f"Executed [bold]{validity['total_attempted']}[/bold] trials "
        f"([green]{validity['total_valid']} valid[/green], [red]{validity['total_invalid']} invalid[/red])."
    )

    console.print(Panel("[bold green]Smoke test PASSED successfully![/bold green]"))


@app.command()
def evaluate(
    data_path: Optional[str] = typer.Option(
        None, "--data", "-d", help="Path to raw HotpotQA JSON file. Defaults to synthetic fixture."
    ),
    manifest_file: Optional[str] = typer.Option(
        None, "--manifest", "-m", help="Path to frozen SplitManifest JSON file."
    ),
    split: str = typer.Option("test", "--split", "-s", help="Target split to evaluate ('dev' | 'test')."),
    regime: str = typer.Option("regime_a", "--regime", "-r", help="Observability regime ('regime_a' | 'regime_b')."),
    output_dir: str = typer.Option("results", "--output", "-o", help="Directory to save output reports."),
    bootstraps: int = typer.Option(100, "--bootstraps", "-b", help="Number of question-level bootstrap iterations."),
    include_f5: bool = typer.Option(False, "--include-f5", help="Include F5 Unsupported Generation in benchmark."),
    include_d3: bool = typer.Option(False, "--include-d3", help="Include D3 NLI Support Diagnostic in benchmark."),
    include_d4_learned: bool = typer.Option(False, "--include-d4-learned", help="Include D4 Learned Decision Tree Policy in benchmark."),
) -> None:
    """Execute full evaluation harness and output metrics summary and confusion matrix."""
    setup_logging(level="INFO")
    console.print(Panel(f"[bold cyan]Running RAG-Autopsy Benchmark ({regime.upper()} on {split.upper()})[/bold cyan]"))

    # Determine data file
    if data_path:
        fpath = Path(data_path)
    else:
        fpath = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "synthetic_qa.json"

    if not fpath.is_file():
        console.print(f"[bold red]Data file not found: {fpath}[/bold red]")
        raise typer.Exit(code=1)

    try:
        records = load_qa_records(fpath)
    except Exception:
        raw_items = load_raw_hotpot_json(fpath)
        records = parse_hotpot_records(raw_items)

    console.print(f"Loaded [cyan]{len(records)}[/cyan] QA records.")

    # Determine or load split manifest
    if manifest_file and Path(manifest_file).is_file():
        from rag_autopsy.data.splits import load_split_manifest
        manifest = load_split_manifest(manifest_file)
        console.print(f"Loaded frozen split manifest: {manifest.split_counts}")
    elif Path("configs/split_manifest_100.json").is_file() and len(records) >= 100:
        from rag_autopsy.data.splits import load_split_manifest
        manifest = load_split_manifest("configs/split_manifest_100.json")
        console.print(f"Loaded existing frozen split manifest: {manifest.split_counts}")
    elif len(records) <= 2:
        manifest = create_question_splits(records, dev_ratio=0.5, calibration_ratio=0.0, test_ratio=0.5, seed=42)
        console.print(f"Partitioned questions across splits: {manifest.split_counts}")
    else:
        manifest = create_question_splits(records, dev_ratio=0.6, calibration_ratio=0.2, test_ratio=0.2, seed=42)
        console.print(f"Partitioned questions across splits: {manifest.split_counts}")

    results = run_experiment_harness(
        qa_records=records,
        split_manifest=manifest,
        target_split=split,
        regime=regime,
        n_bootstraps=bootstraps,
        include_f5=include_f5,
        include_d3=include_d3,
        include_d4_learned=include_d4_learned,
    )

    # Print Results Table
    table = Table(title=f"Diagnostic Attribution Benchmark ({regime}, {split})")
    table.add_column("Diagnostic Method", style="bold yellow")
    table.add_column("Macro-F1 (95% CI)", style="cyan")
    table.add_column("Balanced Acc (95% CI)", style="green")
    table.add_column("Coverage", style="magenta")
    table.add_column("Accuracy (Covered)", style="white")

    for diag_name, data in results["diagnostic_results"].items():
        m = data["metrics"]
        b = data["bootstrap"]
        f1_str = f"{m['macro_f1']:.3f} [{b['macro_f1']['ci_lower']:.3f}, {b['macro_f1']['ci_upper']:.3f}]"
        bal_str = f"{m['balanced_accuracy']:.3f} [{b['balanced_accuracy']['ci_lower']:.3f}, {b['balanced_accuracy']['ci_upper']:.3f}]"
        table.add_row(
            diag_name,
            f1_str,
            bal_str,
            f"{m['coverage']*100:.1f}%",
            f"{m['conditional_accuracy']*100:.1f}%",
        )

    console.print(table)

    # Export machine-readable reports
    artifacts = export_evaluation_report(
        summary_data=results,
        trial_records=results["trial_rows"],
        output_dir=output_dir,
    )
    console.print(f"Saved JSON summary: [bold green]{artifacts['summary_json']}[/bold green]")
    console.print(f"Saved Trial CSV: [bold green]{artifacts['predictions_csv']}[/bold green]")


@app.command()
def audit_natural(
    data_path: Optional[str] = typer.Option(
        None, "--data", "-d", help="Path to QA records JSON file."
    ),
    regime: str = typer.Option("regime_a", "--regime", "-r", help="Observability regime."),
    output_dir: str = typer.Option("results/natural_audit", "--output", "-o", help="Output directory."),
) -> None:
    """Execute natural (unmodified) failure audit and report diagnostic transfer agreement."""
    from rag_autopsy.evaluation.natural_audit import run_natural_failure_audit

    setup_logging(level="INFO")
    console.print(Panel("[bold yellow]Starting Natural Failure Audit on Clean Unmodified Pipeline[/bold yellow]"))

    if data_path:
        fpath = Path(data_path)
    elif Path(".data/processed/hotpot_subset_100.json").is_file():
        fpath = Path(".data/processed/hotpot_subset_100.json")
    else:
        fpath = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "synthetic_qa.json"

    try:
        records = load_qa_records(fpath)
    except Exception:
        raw = load_raw_hotpot_json(fpath)
        records = parse_hotpot_records(raw)

    console.print(f"Loaded [cyan]{len(records)}[/cyan] questions for natural audit.")
    audit_results = run_natural_failure_audit(records, regime=regime)

    # Print summary table
    table = Table(title="Natural Failure Audit & Transfer Analysis")
    table.add_column("Audit Metric", style="bold yellow")
    table.add_column("Value", style="cyan")

    table.add_row("Total Clean Questions Evaluated", str(audit_results["total_evaluated_clean_questions"]))
    table.add_row("Clean Pipeline Accuracy", f"{audit_results['clean_pipeline_accuracy']*100:.1f}%")
    table.add_row("Total Natural Failures Identified", str(audit_results["total_natural_failures"]))
    table.add_row("Natural Failure Rate", f"{audit_results['natural_failure_rate']*100:.1f}%")
    table.add_row("D4 Diagnostic Transfer Agreement", f"[bold green]{audit_results['d4_transfer_agreement_rate']*100:.1f}%[/bold green]")

    console.print(table)

    # Print Cause Breakdown table
    cause_table = Table(title="Adjudicated Natural Failure Causes")
    cause_table.add_column("Cause", style="bold red")
    cause_table.add_column("Count", style="cyan")

    for cause, cnt in audit_results["adjudicated_cause_breakdown"].items():
        cause_table.add_row(cause, str(cnt))

    console.print(cause_table)

    # Save output artifacts
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "natural_audit_summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump(audit_results, f, indent=2)

    console.print(f"Saved natural audit summary: [bold green]{out_dir / 'natural_audit_summary.json'}[/bold green]")


@app.command()
def generate_plots(
    regime_a_summary: str = typer.Option(
        "results/test_regime_a/evaluation_summary.json",
        "--regime-a",
        "-a",
        help="Path to Regime A evaluation summary JSON.",
    ),
    regime_b_summary: Optional[str] = typer.Option(
        "results/test_regime_b/evaluation_summary.json",
        "--regime-b",
        "-b",
        help="Path to Regime B evaluation summary JSON (optional, for side-by-side comparison).",
    ),
    output_dir: str = typer.Option(
        "results/plots",
        "--output-dir",
        "-o",
        help="Directory to save generated PNG plots.",
    ),
) -> None:
    """Generate static publication-quality confusion matrix heatmaps and bootstrap CI plots."""
    from rag_autopsy.reports.plots import generate_all_evaluation_plots

    console.print(Panel("[bold cyan]Generating RAG-Autopsy Evaluation Figures[/bold cyan]"))

    try:
        generated = generate_all_evaluation_plots(
            regime_a_summary_path=regime_a_summary,
            regime_b_summary_path=regime_b_summary,
            output_dir=output_dir,
        )
    except FileNotFoundError as e:
        console.print(f"[bold red]File error:[/bold red] {e}")
        raise typer.Exit(code=1)

    table = Table(title="Generated Figures")
    table.add_column("Figure Description", style="bold yellow")
    table.add_column("Saved File Path", style="green")

    for desc, fpath in generated.items():
        table.add_row(desc, fpath)

    console.print(table)
    console.print(f"[bold green]Successfully generated {len(generated)} plot artifacts in '{output_dir}'.[/bold green]")


@app.command()
def demo(
    data_path: Optional[str] = typer.Option(
        None, "--data", "-d", help="Path to QA records JSON file."
    ),
) -> None:
    """Run interactive/automated 60-second live walkthrough of RAG-Autopsy."""
    from rag_autopsy.demo import run_live_demo

    success = run_live_demo(data_path=data_path, console=console)
    if not success:
        raise typer.Exit(code=1)


@app.command()
def report(
    summary_path: str = typer.Option(
        "results/test_regime_a_extended/evaluation_summary.json",
        "--summary",
        "-s",
        help="Path to evaluation summary JSON.",
    ),
    predictions_csv: str = typer.Option(
        "results/test_regime_a_extended/trial_predictions.csv",
        "--predictions",
        "-p",
        help="Path to trial predictions CSV.",
    ),
    output_path: str = typer.Option(
        "results/report.html",
        "--output",
        "-o",
        help="Path to destination HTML report file.",
    ),
) -> None:
    """Generate standalone interactive HTML report from evaluation results."""
    import csv
    import json
    from rag_autopsy.reports.html_report import generate_interactive_html_report

    s_path = Path(summary_path)
    if not s_path.is_file():
        if Path("results/test_regime_a/evaluation_summary.json").is_file():
            s_path = Path("results/test_regime_a/evaluation_summary.json")
        else:
            console.print(f"[bold red]Evaluation summary not found at: {summary_path}[/bold red]")
            raise typer.Exit(code=1)

    with open(s_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    trials = []
    p_path = Path(predictions_csv)
    if not p_path.is_file() and Path("results/test_regime_a/trial_predictions.csv").is_file():
        p_path = Path("results/test_regime_a/trial_predictions.csv")

    if p_path.is_file():
        with open(p_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            trials = list(reader)

    out_file = generate_interactive_html_report(
        summary_data=summary_data,
        trial_records=trials,
        output_path=output_path,
    )
    console.print(f"[bold green]Successfully generated interactive HTML report: {out_file}[/bold green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

