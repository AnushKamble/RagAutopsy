"""Report serialization: JSON summaries, CSV tables, and Markdown reports."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def export_evaluation_report(
    summary_data: Dict[str, Any],
    trial_records: List[Dict[str, Any]],
    output_dir: str = "results",
) -> Dict[str, str]:
    """Export machine-readable JSON metrics summary and per-trial predictions CSV.
    
    Returns:
        Dict of generated artifact file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / "evaluation_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    csv_file = out_path / "trial_predictions.csv"
    if trial_records:
        fieldnames = list(trial_records[0].keys())
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trial_records)

    return {
        "summary_json": str(json_file),
        "predictions_csv": str(csv_file),
    }
