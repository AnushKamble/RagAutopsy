"""End-to-end RAG reference pipeline orchestrator."""

import time
from typing import List, Optional, Tuple
from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord
from rag_autopsy.evaluation.answer_metrics import evaluate_answer_correctness
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.generator import BaseGenerator
from rag_autopsy.pipeline.retriever import BM25Retriever


class RAGPipeline:
    """Executes retrieval, context assembly, and generation, returning isolated visible and privileged records."""

    def __init__(
        self,
        retriever: BM25Retriever,
        context_builder: Optional[ContextBuilder] = None,
        generator: Optional[BaseGenerator] = None,
        top_k: int = 3,
        candidate_pool_size: int = 20,
    ):
        self.retriever = retriever
        self.context_builder = context_builder or ContextBuilder()
        self.generator = generator
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size

    def run(
        self,
        trial_id: str,
        question_id: str,
        question: str,
        split: str = "dev",
        gold_answers: Optional[List[str]] = None,
        gold_evidence_ids: Optional[List[str]] = None,
        injected_fault: str = "F0",
        fault_parameters: Optional[dict] = None,
        injection_applied: bool = False,
        trace_condition_achieved: bool = True,
        trial_valid: bool = True,
        invalid_reason: Optional[str] = None,
    ) -> Tuple[DiagnosticTrace, EvaluatorRecord]:
        """Execute one query trial end-to-end."""
        start_time = time.perf_counter()

        # 1. Retrieval
        retrieved_passages, candidate_pool_scores = self.retriever.retrieve(
            query=question,
            top_k=self.top_k,
            candidate_pool_size=self.candidate_pool_size,
        )

        # 2. Context assembly
        assembled_context, pre_ids, post_ids = self.context_builder.build_context(retrieved_passages)

        # 3. Generation
        if self.generator:
            answer = self.generator.generate(question, assembled_context)
        else:
            answer = "No generator configured."

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Construct DiagnosticTrace (strictly visible fields only)
        trace = DiagnosticTrace(
            trial_id=trial_id,
            question_id=question_id,
            question=question,
            retrieved_top_k=retrieved_passages,
            candidate_pool_scores=candidate_pool_scores,
            source_passage_ids=[p.passage_id for p in retrieved_passages],
            pre_assembly_passage_ids=pre_ids,
            post_assembly_passage_ids=post_ids,
            generator_context=assembled_context,
            answer=answer,
            latency_ms=round(elapsed_ms, 2),
            generator_model=self.generator.get_model_name() if self.generator else None,
            generator_config=self.generator.get_config() if self.generator else {},
            metadata={
                "top_k": self.top_k,
                "candidate_pool_size": self.candidate_pool_size,
                "max_context_chars": self.context_builder.max_context_chars,
            },
        )

        # 5. Evaluate answer correctness if gold answers are supplied
        em = None
        f1 = None
        if gold_answers:
            em, f1 = evaluate_answer_correctness(answer, gold_answers)

        # 6. Check context support heuristic (whether gold answers appear in assembled context)
        supported = None
        if gold_answers and assembled_context:
            context_lower = assembled_context.lower()
            supported = any(gold.lower() in context_lower for gold in gold_answers)

        # 7. Construct EvaluatorRecord (privileged fields)
        eval_record = EvaluatorRecord(
            trial_id=trial_id,
            question_id=question_id,
            split=split,
            injected_fault=injected_fault,
            fault_parameters=fault_parameters or {},
            gold_answers=gold_answers or [],
            gold_evidence_ids=gold_evidence_ids or [],
            injection_applied=injection_applied,
            trace_condition_achieved=trace_condition_achieved,
            trial_valid=trial_valid,
            invalid_reason=invalid_reason,
            answer_exact_match=em,
            answer_token_f1=f1,
            answer_supported_by_context=supported,
        )

        return trace, eval_record
