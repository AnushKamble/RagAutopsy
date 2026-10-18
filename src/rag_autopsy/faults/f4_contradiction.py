"""F4: Conflicting Evidence injector.

Intervention: Injects a materially contradictory passage into the generator context
alongside the original evidence, so both conflicting claims reach the generator.
"""

from typing import List, Tuple
from rag_autopsy.core.schemas import RetrievedPassage
from rag_autopsy.data.schema import QARecord
from rag_autopsy.faults.base import BaseFaultInjector, InjectionResult
from rag_autopsy.pipeline.corpus import Corpus


class F4ContradictoryEvidenceInjector(BaseFaultInjector):
    """Fault injector for F4: Conflicting Evidence."""

    def __init__(self, fault_id: str = "F4"):
        self.fault_id = fault_id

    def inject(self, qa_record: QARecord, base_corpus: Corpus) -> InjectionResult:
        return InjectionResult(
            injected_fault=self.fault_id,
            fault_parameters={"strategy": "inject_contradiction_in_context"},
            injection_applied=True,
            trace_condition_achieved=True,
            trial_valid=True,
        )

    def apply_to_context_passages(
        self,
        qa_record: QARecord,
        retrieved_passages: List[RetrievedPassage],
    ) -> Tuple[List[RetrievedPassage], InjectionResult]:
        """Inject an explicit contradictory passage into the retrieved list so both reach generator context."""
        gold_passage_ids = set(qa_record.gold_evidence_passage_ids)
        gold_retrieved = [p for p in retrieved_passages if p.passage_id in gold_passage_ids]

        if not gold_retrieved:
            return retrieved_passages, InjectionResult(
                injected_fault=self.fault_id,
                injection_applied=False,
                trace_condition_achieved=False,
                trial_valid=False,
                invalid_reason="Precondition failed: Original gold evidence was not retrieved, so contradiction cannot coexist.",
            )

        # Synthesize an explicit contradictory claim
        contra_id = f"{qa_record.question_id}__f4_contradiction"
        contra_title = f"Contradicting Report on {qa_record.gold_evidence_titles[0] if qa_record.gold_evidence_titles else 'Subject'}"
        contra_text = (
            f"{contra_title}\nOfficial updated records contradict earlier reports: "
            f"The true answer regarding '{qa_record.question}' is emphatically NOT '{qa_record.gold_answer}'. "
            f"Instead, multiple verified sources confirm the exact opposite."
        )

        contradictory_passage = RetrievedPassage(
            passage_id=contra_id,
            rank=len(retrieved_passages) + 1,
            score=9.99,
            text=contra_text,
        )

        modified_passages = list(retrieved_passages) + [contradictory_passage]

        result = InjectionResult(
            injected_fault=self.fault_id,
            fault_parameters={
                "contradictory_passage_id": contra_id,
                "contradicted_gold_answer": qa_record.gold_answer,
            },
            injection_applied=True,
            trace_condition_achieved=True,
            trial_valid=True,
        )
        return modified_passages, result
