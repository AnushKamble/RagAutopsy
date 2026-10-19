"""Combined / Compound fault injector for multi-stage RAG failure simulation.

Enables controlled evaluation of co-occurring faults (e.g. F2 Ranked Out + F3 Assembly Loss,
or F3 Assembly Loss + F4 Contradiction) as specified in Spec Section 6 and Section 12.
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple
from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord, RetrievedPassage
from rag_autopsy.data.schema import PassageRecord, QARecord
from rag_autopsy.evaluation.answer_metrics import evaluate_answer_correctness
from rag_autopsy.faults.base import BaseFaultInjector, InjectionResult
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever, simple_tokenize


class CombinedFaultInjector(BaseFaultInjector):
    """Compound injector that chains multiple single-stage fault interventions.
    
    Supported compositions:
    - F2+F3: Evidence demoted by distractors AND assembly loss drops/truncates surviving evidence.
    - F3+F4: Assembly loss of true evidence AND injection of contradictory claim.
    - F2+F4: Distractors demote evidence AND contradictory passage is introduced.
    """

    def __init__(
        self,
        fault_types: Optional[List[str]] = None,
        distractor_count: int = 3,
        cutoff_k: int = 5,
    ):
        self.fault_types = [f.upper() for f in (fault_types or ["F2", "F3"])]
        self.distractor_count = distractor_count
        self.cutoff_k = cutoff_k
        self.combination_id = "+".join(self.fault_types)

    def inject(self, qa_record: QARecord, base_corpus: Corpus) -> InjectionResult:
        """Apply corpus-level interventions (e.g. F2 distractors)."""
        gold_passage_ids = set(qa_record.gold_evidence_passage_ids)
        if not gold_passage_ids:
            return InjectionResult(
                injected_fault=self.combination_id,
                injection_applied=False,
                trace_condition_achieved=False,
                trial_valid=False,
                invalid_reason="Precondition failed: No gold evidence passage IDs.",
            )

        current_corpus = base_corpus
        applied_stages: List[Dict[str, Any]] = []

        # If F2 is part of combination, inject targeted distractors
        if "F2" in self.fault_types:
            q_tokens = simple_tokenize(qa_record.question)
            if not q_tokens:
                return InjectionResult(
                    injected_fault=self.combination_id,
                    injection_applied=False,
                    trace_condition_achieved=False,
                    trial_valid=False,
                    invalid_reason="Precondition failed: Question has no tokens for F2 distractors.",
                )
            keyword_repeat = " ".join(q_tokens * 3)
            injected_distractors: List[PassageRecord] = []
            for i in range(self.distractor_count):
                d_hash = hashlib.sha256(f"{qa_record.question_id}_cd_{i}".encode("utf-8")).hexdigest()[:8]
                d_id = f"passage_syn_{d_hash}"
                d_title = f"Reference Overview {i+1} regarding {qa_record.question[:20]}"
                d_text = f"{d_title}\nThis document contains background discussion on {keyword_repeat}."
                injected_distractors.append(
                    PassageRecord(
                        passage_id=d_id,
                        title=d_title,
                        text=d_text,
                        sentences=[d_text],
                        is_gold_evidence=False,
                    )
                )
            current_corpus = base_corpus.with_additional_passages(injected_distractors)
            applied_stages.append({
                "stage": "F2",
                "distractors_injected": len(injected_distractors),
                "distractor_ids": [d.passage_id for d in injected_distractors],
            })

        return InjectionResult(
            injected_fault=self.combination_id,
            fault_parameters={"stages": applied_stages},
            injection_applied=True,
            trace_condition_achieved=True,
            trial_valid=True,
            modified_corpus=current_corpus,
        )

    def apply_to_passages(
        self,
        qa_record: QARecord,
        retrieved_passages: List[RetrievedPassage],
    ) -> Tuple[List[RetrievedPassage], InjectionResult]:
        """Apply passage-level interventions (F4 contradiction, F3 assembly loss)."""
        gold_set = set(qa_record.gold_evidence_passage_ids)
        current_passages = list(retrieved_passages)
        stages: List[Dict[str, Any]] = []

        # Check if gold was present in retrieval
        gold_present = any(p.passage_id in gold_set for p in current_passages)

        # Stage: F4 (Contradiction)
        if "F4" in self.fault_types:
            if not gold_present:
                return current_passages, InjectionResult(
                    injected_fault=self.combination_id,
                    injection_applied=False,
                    trace_condition_achieved=False,
                    trial_valid=False,
                    invalid_reason="Precondition failed: Original gold evidence was not retrieved, cannot co-occur with contradiction.",
                )
            contra_hash = hashlib.sha256(f"{qa_record.question_id}_cc_rev".encode("utf-8")).hexdigest()[:8]
            contra_id = f"passage_syn_{contra_hash}"
            subject_title = qa_record.gold_evidence_titles[0] if qa_record.gold_evidence_titles else "Subject"
            contra_title = f"Report on {subject_title}"
            contra_text = (
                f"{contra_title}\nDisputing earlier reports, verified findings establish that the statement regarding "
                f"'{qa_record.question}' contradicts '{qa_record.gold_answer}'. "
                f"The opposite assertion is supported by documented records."
            )
            contra_passage = RetrievedPassage(
                passage_id=contra_id,
                rank=len(current_passages) + 1,
                score=9.99,
                text=contra_text,
            )
            current_passages.append(contra_passage)
            stages.append({"stage": "F4", "contradiction_id": contra_id})

        # Stage: F3 (Assembly Loss)
        if "F3" in self.fault_types:
            if not gold_present:
                return current_passages, InjectionResult(
                    injected_fault=self.combination_id,
                    injection_applied=False,
                    trace_condition_achieved=False,
                    trial_valid=False,
                    invalid_reason="Precondition failed: No gold passage was retrieved in top-k for assembly loss.",
                )
            # Drop gold passages from post-assembly stream
            dropped = [p.passage_id for p in current_passages if p.passage_id in gold_set]
            current_passages = [p for p in current_passages if p.passage_id not in gold_set]
            stages.append({"stage": "F3", "dropped_passage_ids": dropped})

        return current_passages, InjectionResult(
            injected_fault=self.combination_id,
            fault_parameters={"passage_stages": stages},
            injection_applied=True,
            trace_condition_achieved=True,
            trial_valid=True,
        )

    def execute_trial(
        self,
        trial_id: str,
        qa_record: QARecord,
        base_corpus: Corpus,
        pipeline: RAGPipeline,
        split: str = "dev",
    ) -> Tuple[DiagnosticTrace, EvaluatorRecord]:
        """Execute a full pipeline trial with all stages of the compound fault applied."""
        # Step 1: Apply corpus-level interventions
        corpus_res = self.inject(qa_record, base_corpus)
        active_corpus = corpus_res.modified_corpus if corpus_res.modified_corpus is not None else base_corpus

        # Step 2: Retrieve with active corpus
        trial_retriever = BM25Retriever(active_corpus)
        retrieved, pool_scores = trial_retriever.retrieve(
            query=qa_record.question,
            top_k=max(pipeline.top_k, self.cutoff_k),
            candidate_pool_size=pipeline.candidate_pool_size,
        )

        # Step 3: Apply passage/context-level interventions
        post_passages, passage_res = self.apply_to_passages(qa_record, retrieved)

        # Step 4: Assemble generator context
        assembled_ctx, pre_ids, post_ids = pipeline.context_builder.build_context(post_passages)

        # Step 5: Generate answer
        answer = pipeline.generator.generate(qa_record.question, assembled_ctx) if pipeline.generator else ""

        # Step 6: Construct leak-free DiagnosticTrace
        trace = DiagnosticTrace(
            trial_id=trial_id,
            question_id=qa_record.question_id,
            question=qa_record.question,
            retrieved_top_k=retrieved[:pipeline.top_k],
            candidate_pool_scores=pool_scores,
            source_passage_ids=[p.passage_id for p in retrieved[:pipeline.top_k]],
            pre_assembly_passage_ids=pre_ids,
            post_assembly_passage_ids=post_ids,
            generator_context=assembled_ctx,
            answer=answer,
            latency_ms=12.0,
            generator_model=pipeline.generator.get_model_name() if pipeline.generator else None,
            generator_config=pipeline.generator.get_config() if pipeline.generator else {},
            metadata={
                "top_k": pipeline.top_k,
                "candidate_pool_size": pipeline.candidate_pool_size,
                "max_context_chars": pipeline.context_builder.max_context_chars,
            },
        )

        # Step 7: Construct EvaluatorRecord with privileged ground truth
        em, f1 = evaluate_answer_correctness(answer, [qa_record.gold_answer])
        ctx_supported = qa_record.gold_answer.lower() in assembled_ctx.lower() if assembled_ctx else False

        combined_valid = corpus_res.trial_valid and passage_res.trial_valid
        invalid_reasons = []
        if corpus_res.invalid_reason:
            invalid_reasons.append(corpus_res.invalid_reason)
        if passage_res.invalid_reason:
            invalid_reasons.append(passage_res.invalid_reason)

        eval_record = EvaluatorRecord(
            trial_id=trial_id,
            question_id=qa_record.question_id,
            split=split,
            injected_fault=self.combination_id,
            fault_parameters={
                "corpus_stage": corpus_res.fault_parameters,
                "passage_stage": passage_res.fault_parameters,
            },
            gold_answers=[qa_record.gold_answer],
            gold_evidence_ids=qa_record.gold_evidence_passage_ids,
            injection_applied=corpus_res.injection_applied and passage_res.injection_applied,
            trace_condition_achieved=combined_valid,
            trial_valid=combined_valid,
            invalid_reason="; ".join(invalid_reasons) if invalid_reasons else None,
            answer_exact_match=em,
            answer_token_f1=f1,
            answer_supported_by_context=ctx_supported,
        )

        return trace, eval_record
