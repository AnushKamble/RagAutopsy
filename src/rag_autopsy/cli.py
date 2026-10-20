"""Command-line interface for RAG-Autopsy."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_autopsy import __version__
from rag_autopsy.core.config import load_config
from rag_autopsy.core.logging import setup_logging
from rag_autopsy.data.adapter import (
    load_qa_records,
    load_raw_hotpot_json,
    parse_hotpot_records,
)
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
    config_path: str | None = typer.Option(
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
    data_path: str | None = typer.Option(
        None, "--data", "-d", help="Path to raw HotpotQA JSON file. Defaults to synthetic fixture."
    ),
    manifest_file: str | None = typer.Option(
        None, "--manifest", "-m", help="Path to frozen SplitManifest JSON file."
    ),
    split: str = typer.Option("test", "--split", "-s", help="Target split to evaluate ('dev' | 'test')."),
    regime: str = typer.Option("regime_a", "--regime", "-r", help="Observability regime ('regime_a' | 'regime_b')."),
    output_dir: str = typer.Option("results", "--output", "-o", help="Directory to save output reports."),
    bootstraps: int = typer.Option(100, "--bootstraps", "-b", help="Number of question-level bootstrap iterations."),
    generator_type: str = typer.Option("mock", "--generator", "-g", help="Generator strategy ('mock' | 'ollama' | 'hf')."),
    include_f5: bool = typer.Option(False, "--include-f5", help="Include F5 Unsupported Generation in benchmark."),
    include_d3: bool = typer.Option(False, "--include-d3", help="Include D3 NLI Support Diagnostic in benchmark."),
    include_d4_learned: bool = typer.Option(False, "--include-d4-learned", help="Include D4 Learned Decision Tree Policy in benchmark."),
    include_d5: bool = typer.Option(False, "--include-d5", help="Include D5 LLM-as-a-judge diagnostic (Qwen 3 8B via Ollama)."),
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

    # Configure generator
    gen = None
    if generator_type.lower() in ["ollama", "qwen", "qwen3:8b", "qwen2.5:7b"]:
        from rag_autopsy.pipeline.generator import CachedGenerator, OllamaGenerator
        gen = CachedGenerator(OllamaGenerator(model_name="qwen3:8b"))
        console.print("[bold green]Configured local Ollama generator: qwen3:8b (disk cached)[/bold green]")
    elif generator_type.lower() == "hf":
        from rag_autopsy.pipeline.generator import CachedGenerator, HuggingFaceGenerator
        gen = CachedGenerator(HuggingFaceGenerator())
        console.print("[bold green]Configured local HuggingFace generator (disk cached)[/bold green]")

    results = run_experiment_harness(
        qa_records=records,
        split_manifest=manifest,
        target_split=split,
        regime=regime,
        n_bootstraps=bootstraps,
        generator=gen,
        include_f5=include_f5,
        include_d3=include_d3,
        include_d4_learned=include_d4_learned,
        include_d5=include_d5,
    )

    # Print Validity Summary Table
    val = results["validity_summary"]
    val_table = Table(title=f"Fault Injection Validity Accounting ({split.upper()})")
    val_table.add_column("Fault", style="bold yellow")
    val_table.add_column("Attempted", justify="right")
    val_table.add_column("Valid", justify="right", style="green")
    val_table.add_column("Invalid", justify="right", style="red")
    val_table.add_column("Effectiveness", justify="right", style="cyan")
    val_table.add_column("Invalid Reasons", style="magenta")

    for fault_key, fstats in val["by_fault"].items():
        reasons_str = ", ".join(f"{k}: {v}" for k, v in fstats["invalid_reasons"].items()) if fstats["invalid_reasons"] else "None"
        val_table.add_row(
            fault_key,
            str(fstats["attempted"]),
            str(fstats["valid"]),
            str(fstats["invalid"]),
            f"{fstats['effectiveness_rate']*100:.1f}%",
            reasons_str,
        )
    val_table.add_section()
    val_table.add_row(
        "Total",
        str(val["total_attempted"]),
        str(val["total_valid"]),
        str(val["total_invalid"]),
        f"{val['overall_validity_rate']*100:.1f}%",
        "-",
    )
    console.print(val_table)

    # Print Results Table
    table = Table(title=f"Diagnostic Attribution Benchmark ({regime.upper()}, {split.upper()})")
    table.add_column("Diagnostic Method", style="bold yellow")
    table.add_column("Macro-F1 (95% CI)", style="cyan")
    table.add_column("Balanced Acc (95% CI)", style="green")
    table.add_column("Coverage", style="magenta")
    table.add_column("Accuracy (Covered)", style="white")
    table.add_column("Runtime (Total / Avg)", style="yellow")

    for diag_name, data in results["diagnostic_results"].items():
        m = data["metrics"]
        b = data["bootstrap"]
        rt = data.get("runtime", {})
        f1_str = f"{m['macro_f1']:.3f} [{b['macro_f1']['ci_lower']:.3f}, {b['macro_f1']['ci_upper']:.3f}]"
        bal_str = f"{m['balanced_accuracy']:.3f} [{b['balanced_accuracy']['ci_lower']:.3f}, {b['balanced_accuracy']['ci_upper']:.3f}]"
        rt_str = f"{rt.get('total_ms', 0):.1f}ms ({rt.get('avg_per_trial_ms', 0):.2f}ms/t)" if rt else "N/A"
        table.add_row(
            diag_name,
            f1_str,
            bal_str,
            f"{m['coverage']*100:.1f}%",
            f"{m['conditional_accuracy']*100:.1f}%",
            rt_str,
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
    data_path: str | None = typer.Option(
        None, "--data", "-d", help="Path to QA records JSON file."
    ),
    regime: str = typer.Option("regime_a", "--regime", "-r", help="Observability regime."),
    generator_type: str = typer.Option("mock", "--generator", "-g", help="Generator strategy ('mock' | 'ollama' | 'hf')."),
    output_dir: str = typer.Option("results/natural_audit", "--output", "-o", help="Output directory."),
) -> None:
    """Execute natural (unmodified) failure audit and report diagnostic transfer agreement."""
    from rag_autopsy.evaluation.natural_audit import (
        export_natural_audit_report,
        run_natural_failure_audit,
    )

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

    gen = None
    if generator_type.lower() in ["ollama", "qwen", "qwen3:8b", "qwen2.5:7b"]:
        from rag_autopsy.pipeline.generator import CachedGenerator, OllamaGenerator
        gen = CachedGenerator(OllamaGenerator(model_name="qwen3:8b"))
        console.print("[bold green]Configured local Ollama generator: qwen3:8b (disk cached)[/bold green]")
    elif generator_type.lower() == "hf":
        from rag_autopsy.pipeline.generator import CachedGenerator, HuggingFaceGenerator
        gen = CachedGenerator(HuggingFaceGenerator())
        console.print("[bold green]Configured local HuggingFace generator (disk cached)[/bold green]")

    audit_results = run_natural_failure_audit(records, regime=regime, generator=gen)

    # Print summary table
    table = Table(title="Natural Failure Audit & Transfer Analysis")
    table.add_column("Audit Metric", style="bold yellow")
    table.add_column("Value", style="cyan")

    table.add_row("Total Clean Questions Evaluated", str(audit_results["total_evaluated_clean_questions"]))
    table.add_row("Clean Pipeline Accuracy", f"{audit_results['clean_pipeline_accuracy']*100:.1f}%")
    table.add_row("Total Natural Failures Identified", str(audit_results["total_natural_failures"]))
    table.add_row("Natural Failure Rate", f"{audit_results['natural_failure_rate']*100:.1f}%")
    table.add_row("Reviewer Raw Agreement", f"{audit_results['reviewer_agreement']['percent_agreement']*100:.1f}%")
    table.add_row("Reviewer Cohen's Kappa (kappa)", f"{audit_results['reviewer_agreement']['cohen_kappa']:.3f}")
    table.add_row("Total Reviewer Disagreements", str(audit_results['reviewer_agreement']['total_disagreements']))
    table.add_row("D4 Diagnostic Transfer Agreement", f"[bold green]{audit_results['transfer_analysis']['d4_transfer_agreement_rate']*100:.1f}%[/bold green]")

    console.print(table)

    # Print Cause Breakdown table
    cause_table = Table(title="Adjudicated Natural Failure Causes")
    cause_table.add_column("Cause", style="bold red")
    cause_table.add_column("Count", style="cyan")

    for cause, cnt in audit_results["adjudicated_cause_breakdown"].items():
        cause_table.add_row(cause, str(cnt))

    console.print(cause_table)

    # Print Transfer Breakdown by Cause
    transfer_table = Table(title="Diagnostic Transfer Breakdown by Natural Cause")
    transfer_table.add_column("Natural Cause", style="bold yellow")
    transfer_table.add_column("Count", style="white")
    transfer_table.add_column("D4 Recall", style="green")
    transfer_table.add_column("D4 Predictions Distribution", style="magenta")

    for cause, data in audit_results["transfer_analysis"]["breakdown_by_natural_cause"].items():
        dist_str = ", ".join(f"{k}: {v}" for k, v in data["d4_prediction_distribution"].items())
        transfer_table.add_row(cause, str(data["count"]), f"{data['d4_recall']*100:.1f}%", dist_str)

    console.print(transfer_table)

    # Save output artifacts
    artifacts = export_natural_audit_report(audit_results, output_dir=output_dir)
    for k, p in artifacts.items():
        console.print(f"Saved {k}: [bold green]{p}[/bold green]")


@app.command()
def generate_plots(
    regime_a_summary: str = typer.Option(
        "results/test_regime_a/evaluation_summary.json",
        "--regime-a",
        "-a",
        help="Path to Regime A evaluation summary JSON.",
    ),
    regime_b_summary: str | None = typer.Option(
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
    data_path: str | None = typer.Option(
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


def _resolve_records(data_path: str | None):
    if data_path:
        fpath = Path(data_path)
    else:
        fpath = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "synthetic_qa.json"
    if not fpath.is_file():
        raise FileNotFoundError(f"Data file not found: {fpath}")
    try:
        return load_qa_records(fpath)
    except Exception:
        raw = load_raw_hotpot_json(fpath)
        return parse_hotpot_records(raw)


@app.command(name="ablate")
def ablate(
    data_path: str | None = typer.Option(
        None, "--data", "-d", help="Path to QA records JSON file."
    ),
    target_split: str = typer.Option(
        "dev", "--split", "-s", help="Split partition to evaluate ('dev' or 'test')."
    ),
    output_dir: str = typer.Option(
        "results/ablations", "--output-dir", "-o", help="Directory to save ablation report."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
) -> None:
    """Run systematic 5-axis ablation study suite (spec Section 12)."""
    import json

    from rag_autopsy.evaluation.ablations import run_ablation_study_suite

    console.print(Panel("[bold cyan]Running RAG-Autopsy Systematic Ablation Suite[/bold cyan]"))

    records = _resolve_records(data_path)
    console.print(f"Loaded [green]{len(records)}[/green] QA records.")

    manifest = create_question_splits(records, dev_ratio=0.6, calibration_ratio=0.2, test_ratio=0.2, seed=seed)
    report = run_ablation_study_suite(records, manifest, target_split=target_split, seed=seed)

    # Display Diagnostic Ablation
    table1 = Table(title="Ablation 1: Diagnostic Families & Combinations")
    table1.add_column("Diagnostic", style="bold yellow")
    table1.add_column("Macro-F1", style="green")
    table1.add_column("Balanced Accuracy", style="cyan")
    table1.add_column("Coverage", style="white")

    for e in report.diagnostic_ablation:
        table1.add_row(e.diagnostic_name, f"{e.macro_f1:.4f}", f"{e.balanced_accuracy:.4f}", f"{e.coverage:.2%}")
    console.print(table1)

    # Display Abstention Ablation
    table2 = Table(title="Ablation 2: Abstention Policy Tradeoff")
    table2.add_column("Policy Mode", style="bold yellow")
    table2.add_column("Macro-F1", style="green")
    table2.add_column("Diagnosed Accuracy", style="cyan")
    table2.add_column("Coverage", style="white")
    table2.add_column("Abstained", style="red")

    for e in report.abstention_ablation:
        table2.add_row(e.policy_mode, f"{e.macro_f1:.4f}", f"{e.accuracy_on_diagnosed:.4f}", f"{e.coverage:.2%}", str(e.abstained_count))
    console.print(table2)

    # Display Single vs Combined Faults
    table3 = Table(title="Ablation 3: Single vs Compound Fault Attribution")
    table3.add_column("Failure Regime", style="bold yellow")
    table3.add_column("Total Trials", style="white")
    table3.add_column("Macro-F1", style="green")
    table3.add_column("Accuracy", style="cyan")
    table3.add_column("Interpretation", style="italic")

    for e in report.combined_faults_ablation:
        table3.add_row(e.regime_type, str(e.total_trials), f"{e.macro_f1:.4f}", f"{e.accuracy:.4f}", e.notes)
    console.print(table3)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    report_file = out_p / "ablation_study_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    console.print(f"[bold green]Saved ablation study report to: {report_file}[/bold green]")


@app.command(name="severity")
def severity(
    data_path: str | None = typer.Option(
        None, "--data", "-d", help="Path to QA records JSON file."
    ),
    output_dir: str = typer.Option(
        "results/severity", "--output-dir", "-o", help="Directory to save severity report."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
) -> None:
    """Run fault severity sensitivity analysis across distractors and context budget."""
    import json

    from rag_autopsy.evaluation.severity import run_severity_analysis
    from rag_autopsy.pipeline.corpus import Corpus

    console.print(Panel("[bold cyan]Running RAG-Autopsy Fault Severity Sensitivity Analysis[/bold cyan]"))

    records = _resolve_records(data_path)
    base_corpus = Corpus.from_qa_records(records)

    report = run_severity_analysis(
        records=records,
        base_corpus=base_corpus,
        distractor_counts=[1, 3, 5, 10, 20],
        context_budgets=[500, 1000, 2000, 3000],
        seed=seed,
    )

    t_dist = Table(title="Distractor Count Severity Sweep (F2 Ranked Out)")
    t_dist.add_column("Distractor Count", style="bold yellow")
    t_dist.add_column("Validity Rate", style="cyan")
    t_dist.add_column("D1 Recall", style="green")
    t_dist.add_column("D4 Recall", style="magenta")
    t_dist.add_column("Mean Token F1", style="white")

    for pt in report.distractor_curve:
        t_dist.add_row(
            str(pt.distractor_count),
            f"{pt.validity_rate:.2%}",
            f"{pt.d1_f2_recall:.4f}",
            f"{pt.d4_f2_recall:.4f}",
            f"{pt.mean_token_f1:.4f}",
        )
    console.print(t_dist)

    t_ctx = Table(title="Context Budget Severity Sweep (F3 Assembly Loss)")
    t_ctx.add_column("Max Context Chars", style="bold yellow")
    t_ctx.add_column("Validity Rate", style="cyan")
    t_ctx.add_column("Retention Rate", style="green")
    t_ctx.add_column("D2 Recall", style="magenta")
    t_ctx.add_column("D4 Recall", style="yellow")

    for pt in report.context_budget_curve:
        t_ctx.add_row(
            str(pt.max_context_chars),
            f"{pt.validity_rate:.2%}",
            f"{pt.evidence_retention_rate:.2%}",
            f"{pt.d2_f3_recall:.4f}",
            f"{pt.d4_f3_recall:.4f}",
        )
    console.print(t_ctx)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_file = out_p / "severity_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    console.print(f"[bold green]Saved severity sensitivity report to: {out_file}[/bold green]")


@app.command(name="ragtruth")
def ragtruth(
    fixture_path: str | None = typer.Option(
        None, "--file", "-f", help="Path to RAGTruth JSON samples file."
    ),
    output_dir: str = typer.Option(
        "results/ragtruth", "--output-dir", "-o", help="Directory to save RAGTruth report."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
) -> None:
    """Run hallucination/grounding benchmark on RAGTruth human-annotated dataset."""
    import json

    from rag_autopsy.data.ragtruth import load_ragtruth_json
    from rag_autopsy.diagnostics.d2_support import D2LexicalSupportDiagnostic
    from rag_autopsy.diagnostics.d3_nli import D3NLISupportDiagnostic
    from rag_autopsy.evaluation.ragtruth_eval import run_ragtruth_benchmark

    console.print(Panel("[bold cyan]Running RAGTruth Grounding & Hallucination Benchmark[/bold cyan]"))

    default_fixture = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "ragtruth_synthetic.json"
    p = Path(fixture_path) if fixture_path else default_fixture

    samples = load_ragtruth_json(p)
    console.print(f"Loaded [green]{len(samples)}[/green] RAGTruth samples from: {p}")

    d2 = D2LexicalSupportDiagnostic()
    d3 = D3NLISupportDiagnostic()

    report = run_ragtruth_benchmark(samples, [d2, d3], seed=seed)

    table = Table(title="RAGTruth Diagnostic Benchmark (Agreement with Human Annotations)")
    table.add_column("Diagnostic", style="bold yellow")
    table.add_column("Accuracy", style="green")
    table.add_column("Balanced Acc", style="cyan")
    table.add_column("Precision", style="white")
    table.add_column("Recall", style="magenta")
    table.add_column("F1 Score", style="yellow")
    table.add_column("Mean Latency", style="blue")

    for name, m in report.diagnostic_metrics.items():
        table.add_row(
            m.diagnostic_name,
            f"{m.accuracy:.4f}",
            f"{m.balanced_accuracy:.4f}",
            f"{m.precision:.4f}",
            f"{m.recall:.4f}",
            f"{m.f1_score:.4f}",
            f"{m.mean_latency_ms:.2f} ms",
        )
    console.print(table)
    console.print(f"[italic]{report.comparative_summary}[/italic]")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_file = out_p / "ragtruth_benchmark_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    console.print(f"[bold green]Saved RAGTruth report to: {out_file}[/bold green]")


@app.command(name="quality-compare")
def quality_compare(
    data_path: str | None = typer.Option(
        None, "--data", "-d", help="Path to QA records JSON file."
    ),
    target_split: str = typer.Option(
        "dev", "--split", "-s", help="Target split to evaluate."
    ),
    output_dir: str = typer.Option(
        "results/rag_quality", "--output-dir", "-o", help="Directory to save quality report."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
) -> None:
    """Compare conventional scalar RAG quality metrics (Ragas) vs RAG-Autopsy fault attribution."""
    import json

    from rag_autopsy.evaluation.rag_quality import evaluate_rag_quality_dataset

    console.print(Panel("[bold cyan]Comparing Scalar RAG Quality Metrics vs Fault Attribution[/bold cyan]"))

    records = _resolve_records(data_path)
    manifest = create_question_splits(records, dev_ratio=0.6, calibration_ratio=0.2, test_ratio=0.2, seed=seed)

    harness = run_experiment_harness(
        qa_records=records,
        split_manifest=manifest,
        target_split=target_split,
        include_f5=True,
        include_d3=True,
        seed=seed,
    )

    traces = harness["traces"]
    eval_records = harness["evaluator_records"]

    comp = evaluate_rag_quality_dataset(traces, eval_records)

    console.print(
        f"Overall Metrics across [bold]{comp.total_trials}[/bold] trials: "
        f"Context Precision@k=[cyan]{comp.overall_context_precision:.4f}[/cyan], "
        f"Faithfulness=[green]{comp.overall_faithfulness:.4f}[/green], "
        f"Token F1=[yellow]{comp.overall_token_f1:.4f}[/yellow]"
    )

    table = Table(title="RAG Quality Breakdown by Fault Class vs Attribution Isolation")
    table.add_column("Injected Fault", style="bold yellow")
    table.add_column("Trials", style="white")
    table.add_column("Context Prec@k", style="cyan")
    table.add_column("Faithfulness", style="green")
    table.add_column("Token F1", style="yellow")
    table.add_column("Attribution Distinction (Why Scalar Metrics Fail)", style="italic")

    for fid, prof in comp.profiles_by_fault.items():
        table.add_row(
            fid,
            str(prof.trial_count),
            f"{prof.mean_context_precision:.4f}",
            f"{prof.mean_faithfulness:.4f}",
            f"{prof.mean_token_f1:.4f}",
            prof.attribution_distinction,
        )
    console.print(table)
    console.print(f"[bold cyan]{comp.interpretability_gap_summary}[/bold cyan]")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_file = out_p / "quality_vs_attribution.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comp.model_dump(), f, indent=2)

    console.print(f"[bold green]Saved comparison to: {out_file}[/bold green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

