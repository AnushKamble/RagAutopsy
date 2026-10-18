"""Standalone script to execute held-out injected evaluation (Regime A and Regime B)."""

import argparse
from pathlib import Path
from rag_autopsy.data.adapter import load_qa_records, load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.data.splits import create_question_splits, load_split_manifest
from rag_autopsy.evaluation.runner import run_experiment_harness
from rag_autopsy.reports.summary import export_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Run RAG-Autopsy injected evaluation benchmark.")
    parser.add_argument("--data", type=str, default=None, help="Path to QA data JSON.")
    parser.add_argument("--split", type=str, default="test", help="Target split (dev|test).")
    parser.add_argument("--regime", type=str, default="regime_a", help="Regime (regime_a|regime_b).")
    parser.add_argument("--output-dir", type=str, default="results/injected_eval", help="Output directory.")
    parser.add_argument("--bootstraps", type=int, default=100, help="Bootstrap iterations.")
    args = parser.parse_args()

    if args.data:
        fpath = Path(args.data)
    else:
        fpath = Path("tests/fixtures/synthetic_qa.json")

    try:
        records = load_qa_records(fpath)
    except Exception:
        records = parse_hotpot_records(load_raw_hotpot_json(fpath))

    manifest_path = Path("configs/split_manifest_100.json")
    if manifest_path.is_file() and len(records) >= 100:
        manifest = load_split_manifest(manifest_path)
    else:
        manifest = create_question_splits(records, dev_ratio=0.6, calibration_ratio=0.2, test_ratio=0.2, seed=42)

    print(f"Running evaluation on {len(records)} records ({args.regime.upper()} on {args.split.upper()})...")
    harness_results = run_experiment_harness(
        qa_records=records,
        split_manifest=manifest,
        target_split=args.split,
        regime=args.regime,
        n_bootstraps=args.bootstraps,
        include_f5=True,
        include_d3=True,
        include_d4_learned=True,
    )

    out_p = Path(args.output_dir)
    export_evaluation_report(
        results=harness_results,
        output_summary_path=out_p / "evaluation_summary.json",
        output_csv_path=out_p / "trial_predictions.csv",
    )
    print(f"Evaluation complete. Artifacts written to: {out_p}")


if __name__ == "__main__":
    main()
