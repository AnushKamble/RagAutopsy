"""F3: Context Assembly Loss injector.

Intervention: Gold evidence is retrieved in top-k, but is truncated/dropped
during context assembly, so it never reaches the generator.
"""


from rag_autopsy.core.schemas import RetrievedPassage
from rag_autopsy.data.schema import QARecord
from rag_autopsy.faults.base import BaseFaultInjector, InjectionResult
from rag_autopsy.pipeline.corpus import Corpus


class F3ContextAssemblyLossInjector(BaseFaultInjector):
    """Fault injector for F3: Context Assembly Loss."""

    def __init__(self, fault_id: str = "F3"):
        self.fault_id = fault_id

    def inject(self, qa_record: QARecord, base_corpus: Corpus) -> InjectionResult:
        """Inject F3 at the interface between retriever and context builder."""
        return InjectionResult(
            injected_fault=self.fault_id,
            fault_parameters={"strategy": "drop_gold_from_context"},
            injection_applied=True,
            trace_condition_achieved=True,
            trial_valid=True,
        )

    def apply_to_retrieved_passages(
        self,
        retrieved_passages: list[RetrievedPassage],
        gold_passage_ids: list[str],
    ) -> tuple[list[RetrievedPassage], InjectionResult]:
        """Filter out gold passages from the post-assembly context stream.

        Returns:
            Tuple of:
            - modified_passages: list of passages that survive into context assembly
            - injection_result: InjectionResult
        """
        gold_set: set[str] = set(gold_passage_ids)
        gold_present_in_retrieved = any(p.passage_id in gold_set for p in retrieved_passages)

        if not gold_present_in_retrieved:
            # Ineffective injection: gold was never retrieved, so context assembly loss cannot occur!
            return retrieved_passages, InjectionResult(
                injected_fault=self.fault_id,
                injection_applied=False,
                trace_condition_achieved=False,
                trial_valid=False,
                invalid_reason="Precondition failed: No gold passage was retrieved in top-k.",
            )

        # Drop gold passages from post-assembly stream
        surviving_passages = [p for p in retrieved_passages if p.passage_id not in gold_set]

        res = InjectionResult(
            injected_fault=self.fault_id,
            fault_parameters={"dropped_passage_ids": [p.passage_id for p in retrieved_passages if p.passage_id in gold_set]},
            injection_applied=True,
            trace_condition_achieved=True,
            trial_valid=True,
        )
        return surviving_passages, res
