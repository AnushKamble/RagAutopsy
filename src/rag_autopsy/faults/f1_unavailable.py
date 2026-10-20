"""F1: Evidence Unavailable injector.

Intervention: Removes all gold supporting passages from the searchable corpus
for that trial, simulating indexing failure or document omission.
"""

from rag_autopsy.data.schema import QARecord
from rag_autopsy.faults.base import BaseFaultInjector, InjectionResult
from rag_autopsy.pipeline.corpus import Corpus


class F1EvidenceUnavailableInjector(BaseFaultInjector):
    """Fault injector for F1: Evidence Unavailable."""

    def __init__(self, fault_id: str = "F1"):
        self.fault_id = fault_id

    def inject(self, qa_record: QARecord, base_corpus: Corpus) -> InjectionResult:
        gold_passage_ids: set[str] = set(qa_record.gold_evidence_passage_ids)

        if not gold_passage_ids:
            return InjectionResult(
                injected_fault=self.fault_id,
                fault_parameters={},
                injection_applied=False,
                trace_condition_achieved=False,
                trial_valid=False,
                invalid_reason="Precondition failed: QARecord contains no gold evidence passage IDs.",
            )

        # Check if gold passages actually exist in the base corpus
        existing_gold = [pid for pid in gold_passage_ids if base_corpus.get_passage(pid) is not None]
        if not existing_gold:
            return InjectionResult(
                injected_fault=self.fault_id,
                fault_parameters={"target_gold_ids": list(gold_passage_ids)},
                injection_applied=False,
                trace_condition_achieved=False,
                trial_valid=False,
                invalid_reason="Precondition failed: Gold passages already absent from base corpus.",
            )

        # Apply intervention: produce trial-scoped corpus without gold passages
        trial_corpus = base_corpus.without_passages(gold_passage_ids)

        # Verify trace condition: gold passages are completely absent from trial corpus
        remaining_gold = [pid for pid in gold_passage_ids if trial_corpus.get_passage(pid) is not None]
        condition_met = (len(remaining_gold) == 0)

        result = InjectionResult(
            injected_fault=self.fault_id,
            fault_parameters={
                "removed_passage_ids": existing_gold,
                "original_corpus_size": len(base_corpus),
                "modified_corpus_size": len(trial_corpus),
            },
            injection_applied=True,
            trace_condition_achieved=condition_met,
            trial_valid=condition_met,
        )
        # Store trial_corpus on the result object for pipeline consumption
        result.modified_corpus = trial_corpus  # type: ignore[attr-defined]
        return result
