"""F2: Evidence Ranked Out injector.

Intervention: Injects high-scoring query-targeted distractors into the searchable corpus
so that gold evidence remains searchable, but drops below top-k (rank > k).
"""

from typing import List, Set
from rag_autopsy.data.schema import PassageRecord, QARecord
from rag_autopsy.faults.base import BaseFaultInjector, InjectionResult
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.retriever import BM25Retriever, simple_tokenize


class F2EvidenceRankedOutInjector(BaseFaultInjector):
    """Fault injector for F2: Evidence Ranked Out."""

    def __init__(self, fault_id: str = "F2", distractor_count: int = 5, cutoff_k: int = 3):
        self.fault_id = fault_id
        self.distractor_count = distractor_count
        self.cutoff_k = cutoff_k

    def inject(self, qa_record: QARecord, base_corpus: Corpus) -> InjectionResult:
        gold_passage_ids: Set[str] = set(qa_record.gold_evidence_passage_ids)

        if not gold_passage_ids:
            return InjectionResult(
                injected_fault=self.fault_id,
                injection_applied=False,
                trace_condition_achieved=False,
                trial_valid=False,
                invalid_reason="Precondition failed: No gold evidence passage IDs.",
            )

        # Synthesize distractor passages loaded with question terms to outrank gold passages
        q_tokens = simple_tokenize(qa_record.question)
        if not q_tokens:
            return InjectionResult(
                injected_fault=self.fault_id,
                injection_applied=False,
                trace_condition_achieved=False,
                trial_valid=False,
                invalid_reason="Precondition failed: Question has no tokens.",
            )

        # Create distracting passages repeating question keywords heavily to inflate BM25 score
        keyword_repeat = " ".join(q_tokens * 3)
        injected_distractors: List[PassageRecord] = []

        for i in range(self.distractor_count):
            d_id = f"{qa_record.question_id}__f2_distractor_{i+1}"
            d_title = f"Distractor Overview {i+1} regarding {qa_record.question[:20]}"
            d_text = f"{d_title}\nThis document contains background discussion on {keyword_repeat}. It mentions various unrelated historical events."
            injected_distractors.append(
                PassageRecord(
                    passage_id=d_id,
                    title=d_title,
                    text=d_text,
                    sentences=[d_text],
                    is_gold_evidence=False,
                )
            )

        trial_corpus = base_corpus.with_additional_passages(injected_distractors)

        # Verify state change under BM25:
        retriever = BM25Retriever(trial_corpus)
        retrieved, candidate_scores = retriever.retrieve(
            query=qa_record.question, top_k=self.cutoff_k, candidate_pool_size=20
        )
        top_k_ids = [p.passage_id for p in retrieved]

        # Condition achieved if none of the gold passage IDs appear in top_k_ids,
        # but they still exist in the trial corpus
        gold_in_top_k = any(gid in top_k_ids for gid in gold_passage_ids)
        gold_still_in_corpus = all(trial_corpus.get_passage(gid) is not None for gid in gold_passage_ids)

        condition_met = (not gold_in_top_k) and gold_still_in_corpus

        result = InjectionResult(
            injected_fault=self.fault_id,
            fault_parameters={
                "distractor_count": self.distractor_count,
                "cutoff_k": self.cutoff_k,
                "injected_distractor_ids": [d.passage_id for d in injected_distractors],
            },
            injection_applied=True,
            trace_condition_achieved=condition_met,
            trial_valid=condition_met,
            invalid_reason=None if condition_met else "Gold passage still ranked in top-k despite distractors.",
        )
        result.modified_corpus = trial_corpus  # type: ignore[attr-defined]
        return result
