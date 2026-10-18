"""Pilot experiment runner verifying all injectors and trace generation."""

from pathlib import Path
from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.data.splits import create_question_splits
from rag_autopsy.evaluation.runner import run_experiment_harness


def main():
    fixture_path = Path("tests/fixtures/synthetic_qa.json")
    records = parse_hotpot_records(load_raw_hotpot_json(fixture_path))
    manifest = create_question_splits(records, dev_ratio=0.5, calibration_ratio=0.0, test_ratio=0.5, seed=42)

    print(f"Running multi-fault pilot on {len(records)} QA records...")
    results = run_experiment_harness(
        qa_records=records,
        split_manifest=manifest,
        target_split="dev",
        include_f5=True,
        include_d3=True,
        n_bootstraps=10,
        seed=42,
    )

    validity = results["validity_summary"]
    print(f"Pilot execution completed: {validity['total_attempted']} attempted, "
          f"{validity['total_valid']} valid, {validity['total_invalid']} invalid.")

    print("\nDiagnostic Macro-F1 Results:")
    for name, data in results["diagnostic_results"].items():
        m = data["metrics"]
        print(f"  - {name}: Macro-F1 = {m['macro_f1']:.4f}, Coverage = {m['coverage']:.2%}")


if __name__ == "__main__":
    main()
