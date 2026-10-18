"""Injection validity adjudication and effectiveness accounting."""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel

from rag_autopsy.core.schemas import EvaluatorRecord


class FaultValidityStats(BaseModel):
    attempted: int
    valid: int
    invalid: int
    effectiveness_rate: float
    invalid_reasons: Dict[str, int]


class InjectionValiditySummary(BaseModel):
    total_attempted: int
    total_valid: int
    total_invalid: int
    overall_validity_rate: float
    by_fault: Dict[str, FaultValidityStats]


def adjudicate_trials(
    evaluator_records: List[EvaluatorRecord],
) -> Tuple[List[EvaluatorRecord], List[EvaluatorRecord], InjectionValiditySummary]:
    """Separate valid trials from invalid trials and generate a full validity accounting report.
    
    Returns:
        (valid_records, invalid_records, validity_summary)
    """
    valid_records: List[EvaluatorRecord] = []
    invalid_records: List[EvaluatorRecord] = []

    counts_by_fault = defaultdict(lambda: {"attempted": 0, "valid": 0, "invalid": 0, "reasons": Counter()})

    for rec in evaluator_records:
        fault = rec.injected_fault
        counts_by_fault[fault]["attempted"] += 1

        if rec.trial_valid:
            valid_records.append(rec)
            counts_by_fault[fault]["valid"] += 1
        else:
            invalid_records.append(rec)
            counts_by_fault[fault]["invalid"] += 1
            reason = rec.invalid_reason or "Unspecified validity failure"
            counts_by_fault[fault]["reasons"][reason] += 1

    total_attempted = len(evaluator_records)
    total_valid = len(valid_records)
    total_invalid = len(invalid_records)

    by_fault_stats: Dict[str, FaultValidityStats] = {}
    for fault, data in sorted(counts_by_fault.items()):
        att = data["attempted"]
        val = data["valid"]
        eff = round(val / att, 4) if att > 0 else 0.0
        by_fault_stats[fault] = FaultValidityStats(
            attempted=att,
            valid=val,
            invalid=data["invalid"],
            effectiveness_rate=eff,
            invalid_reasons=dict(data["reasons"]),
        )

    summary = InjectionValiditySummary(
        total_attempted=total_attempted,
        total_valid=total_valid,
        total_invalid=total_invalid,
        overall_validity_rate=round(total_valid / total_attempted, 4) if total_attempted > 0 else 0.0,
        by_fault=by_fault_stats,
    )
    return valid_records, invalid_records, summary
