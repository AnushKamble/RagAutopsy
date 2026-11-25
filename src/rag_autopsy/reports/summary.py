"""Report serialization: JSON summaries, CSV tables, and separate JSONL trace/evaluator files."""

import csv
import json
from pathlib import Path
from typing import Any

from rag_autopsy.core.schemas import (
    DiagnosticTrace,
    EvaluatorRecord,
    save_evaluator_records_jsonl,
    save_traces_jsonl,
)


def export_evaluation_report(
    summary_data: dict[str, Any],
    trial_records: list[dict[str, Any]],
    output_dir: str = "results",
    traces: list[DiagnosticTrace] | None = None,
    evaluator_records: list[EvaluatorRecord] | None = None,
) -> dict[str, str]:
    """Export machine-readable JSON metrics summary, per-trial predictions CSV,
    and strictly decoupled diagnostic_traces.jsonl and evaluator_records.jsonl files.

    Returns:
        Dict of generated artifact file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Extract traces and evaluator store if present in summary_data to avoid json serialization error
    clean_summary = dict(summary_data)
    extracted_traces = traces or clean_summary.pop("traces", None)
    eval_store = clean_summary.pop("evaluator_store", None)
    extracted_eval_records = evaluator_records or clean_summary.pop("evaluator_records", None)

    json_file = out_path / "evaluation_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)

    csv_file = out_path / "trial_predictions.csv"
    if trial_records:
        fieldnames = list(trial_records[0].keys())
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trial_records)

    artifacts = {
        "summary_json": str(json_file),
        "predictions_csv": str(csv_file),
    }

    # Save physical separate file boundaries
    if extracted_traces:
        traces_file = out_path / "diagnostic_traces.jsonl"
        save_traces_jsonl(extracted_traces, traces_file)
        artifacts["traces_jsonl"] = str(traces_file)

    if eval_store:
        eval_file = out_path / "evaluator_records.jsonl"
        eval_store.to_jsonl(eval_file)
        artifacts["evaluator_jsonl"] = str(eval_file)
    elif extracted_eval_records:
        eval_file = out_path / "evaluator_records.jsonl"
        save_evaluator_records_jsonl(extracted_eval_records, eval_file)
        artifacts["evaluator_jsonl"] = str(eval_file)

    return artifacts

