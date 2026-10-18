"""Standalone script to execute natural failure audit on unmodified QA trials."""

import argparse
import json
from pathlib import Path
from rag_autopsy.data.adapter import load_qa_records, load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.evaluation.natural_audit import run_natural_failure_audit


def main():
    parser = argparse.ArgumentParser(description="Run RAG-Autopsy natural failure audit.")
    parser.add_argument("--data", type=str, default=None, help="Path to QA data JSON.")
    parser.add_argument("--output-dir", type=str, default="results/natural_audit", help="Output directory.")
    args = parser.parse_args()

    if args.data:
        fpath = Path(args.data)
    elif Path(".data/processed/hotpot_subset_100.json").is_file():
        fpath = Path(".data/processed/hotpot_subset_100.json")
    else:
        fpath = Path("tests/fixtures/synthetic_qa.json")

    try:
        records = load_qa_records(fpath)
    except Exception:
        records = parse_hotpot_records(load_raw_hotpot_json(fpath))

    print(f"Auditing natural failures across {len(records)} unmodified QA records...")
    audit_results = run_natural_failure_audit(records)

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_file = out_p / "natural_audit_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"Audit completed: {audit_results['total_natural_failures']} natural failures identified.")
    print(f"Audit report saved to: {out_file}")


if __name__ == "__main__":
    main()
