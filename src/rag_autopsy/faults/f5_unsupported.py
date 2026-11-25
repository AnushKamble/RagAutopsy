"""F5: Unsupported Generation / Hallucination injector.

Intervention: Gold evidence is successfully retrieved and present in the generator context,
but the generator emits an unsupported/hallucinated answer that is not grounded in the context.
"""

from rag_autopsy.core.schemas import RetrievedPassage
from rag_autopsy.data.schema import QARecord
from rag_autopsy.faults.base import BaseFaultInjector, InjectionResult
from rag_autopsy.pipeline.corpus import Corpus


class F5UnsupportedGenerationInjector(BaseFaultInjector):
    """Fault injector for F5: Unsupported Generation despite adequate context."""

    def __init__(self, fault_id: str = "F5"):
        self.fault_id = fault_id

    def inject(self, qa_record: QARecord, base_corpus: Corpus) -> InjectionResult:
        """Record F5 intervention metadata."""
        return InjectionResult(
            injected_fault=self.fault_id,
            fault_parameters={
                "strategy": "induce_unsupported_hallucination",
                "gold_answer": qa_record.gold_answer,
            },
            injection_applied=True,
            trace_condition_achieved=True,
            trial_valid=True,
        )

    def apply_to_generation(
        self,
        qa_record: QARecord,
        retrieved_passages: list[RetrievedPassage],
        assembled_context: str,
    ) -> tuple[str, InjectionResult]:
        """Verify evidence is present in context, and generate an unsupported answer.

        Returns:
            Tuple of:
            - unsupported_answer: string claim not supported by assembled_context
            - injection_result: InjectionResult detailing validity adjudication
        """
        # Precondition 1: Check if any gold evidence passage was retrieved
        gold_ids = set(qa_record.gold_evidence_passage_ids)
        retrieved_ids = {p.passage_id for p in retrieved_passages}
        gold_retrieved = len(gold_ids.intersection(retrieved_ids)) > 0

        # Precondition 2: Check if gold evidence or gold answer text appears in context
        evidence_in_context = any(p.passage_id in gold_ids for p in retrieved_passages)

        if not gold_retrieved or not evidence_in_context:
            return (
                "Unsupported answer (ineffective: evidence was absent from context)",
                InjectionResult(
                    injected_fault=self.fault_id,
                    injection_applied=False,
                    trace_condition_achieved=False,
                    trial_valid=False,
                    invalid_reason="Precondition failed: Gold evidence was not retrieved or present in context, so failure cannot be isolated to generator hallucination.",
                ),
            )

        # Generate a plausible but strictly unsupported counter-factual answer
        # e.g., substitute numbers, names, or locations with fictitious variants
        gold_ans = qa_record.gold_answer
        hallucinated_answer = f"According to historical records, the answer is definitively {gold_ans}_UNKNOWN_FICTITIOUS_VARIANT_1999."

        # If gold answer was a year or number, perturb it
        import re
        numbers = re.findall(r"\b\d+\b", gold_ans)
        if numbers:
            perturbed = gold_ans
            for num in numbers:
                perturbed = perturbed.replace(num, str(int(num) + 17))
            hallucinated_answer = f"The official documented figure is {perturbed}."
        elif len(gold_ans.split()) <= 4:
            hallucinated_answer = f"The documented subject is {gold_ans} (disputed by Alternative Archive Records)."
        else:
            hallucinated_answer = f"The subject was reported to be completely unrelated to {gold_ans[:20]}."

        res = InjectionResult(
            injected_fault=self.fault_id,
            fault_parameters={
                "gold_answer": gold_ans,
                "hallucinated_answer": hallucinated_answer,
                "evidence_present_in_context": True,
            },
            injection_applied=True,
            trace_condition_achieved=True,
            trial_valid=True,
        )
        return hallucinated_answer, res
