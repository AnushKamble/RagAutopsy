"""Standalone script to execute the 5-axis systematic ablation study suite."""

import argparse
import json
from pathlib import Path
from rag_autopsy.data.adapter import load_qa_records, load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.data.splits import create_question_splits, load_split_manifest
from rag_autopsy.evaluation.ablations import run_ablation_study_suite


def main():
    parser = argparse.ArgumentParser(description="Run RAG-Autopsy 5-axis systematic ablation study.")
    parser.add_argument("--data", type=str, default=None, help="Path to QA data JSON.")
    parser.add_argument("--split", type=str, default="dev", help="Split partition to evaluate.")
    parser.add_argument("--output-dir", type=str, default="results/ablations", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
        manifest = create_question_splits(records, dev_ratio=0.6, calibration_ratio=0.2, test_ratio=0.2, seed=args.seed)

    print(f"Running 5-axis ablation study suite on {len(records)} records (split: {args.split})...")
    report = run_ablation_study_suite(records, manifest, target_split=args.split, seed=args.seed)

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_file = out_p / "ablation_study_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    print(f"Ablation study complete. Report saved to: {out_file}")


if __name__ == "__main__":
    main()
