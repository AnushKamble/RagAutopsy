"""Natural failure audit engine: analyzing transfer of diagnostics to organic pipeline errors."""

from collections import Counter
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord
from rag_autopsy.data.schema import QARecord
from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic
from rag_autopsy.diagnostics.d2_support import D2LexicalSupportDiagnostic
from rag_autopsy.diagnostics.d4_policy import D4CombinedAttributionPolicy
from rag_autopsy.evaluation.answer_metrics import evaluate_answer_correctness
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import BaseGenerator, MockExtractiveGenerator
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever


class NaturalFailureRecord(BaseModel):
    question_id: str
    question: str
    gold_answer: str
    generated_answer: str
    retrieved_passage_ids: List[str]
    context_passage_ids: List[str]
    adjudicated_cause: str
    d1_prediction: str
    d2_prediction: str
    d4_prediction: str
    d4_abstained: bool
    d4_rationale: str


def run_natural_failure_audit(
    qa_records: List[QARecord],
    generator: Optional[BaseGenerator] = None,
    top_k: int = 3,
    candidate_pool_size: int = 20,
    regime: str = "regime_a",
) -> Dict[str, Any]:
    """Execute clean RAG pipeline across questions, extract natural failures, and evaluate transfer."""
    corpus = Corpus.from_qa_records(qa_records)
    retriever = BM25Retriever(corpus)
    builder = ContextBuilder(max_context_chars=3000)
    gen = generator or MockExtractiveGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=builder,
        generator=gen,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
    )

    d1 = D1RetrievalDiagnostic(regime=regime)
    d2 = D2LexicalSupportDiagnostic()
    d4 = D4CombinedAttributionPolicy(regime=regime)

    total_evaluated = len(qa_records)
    natural_failures: List[NaturalFailureRecord] = []
    cause_counts = Counter()

    for idx, rec in enumerate(qa_records):
        trial_id = f"natural_trial_{idx+1:04d}"
        trace, eval_rec = pipeline.run(
            trial_id=trial_id,
            question_id=rec.question_id,
            question=rec.question,
            split="natural_audit",
            gold_answers=[rec.gold_answer],
            gold_evidence_ids=rec.gold_evidence_passage_ids,
            injected_fault="F0",  # Clean run: NO intentional injection
        )

        # Natural failure condition: exact match is False
        if not eval_rec.answer_exact_match:
            # Determine objective cause based on the annotation rubric
            gold_ids = set(rec.gold_evidence_passage_ids)
            retrieved_ids = set(trace.pre_assembly_passage_ids)
            context_ids = set(trace.post_assembly_passage_ids)

            gold_retrieved = len(gold_ids.intersection(retrieved_ids)) > 0
            gold_in_context = len(gold_ids.intersection(context_ids)) > 0

            if not gold_retrieved:
                # Evidence was missed during retrieval
                # Check candidate pool
                pool_passages, _ = retriever.retrieve(rec.question, top_k=candidate_pool_size)
                pool_ids = set(p.passage_id for p in pool_passages)
                if len(gold_ids.intersection(pool_ids)) > 0:
                    adjudicated = "F2"  # Ranked out
                else:
                    adjudicated = "F1"  # Unavailable in pool
            elif gold_retrieved and not gold_in_context:
                adjudicated = "F3"  # Assembly loss
            else:
                # Evidence reached generator context, but answer failed
                adjudicated = "REASONING_ERROR"

            cause_counts[adjudicated] += 1

            diag_1 = d1.diagnose(trace)
            diag_2 = d2.diagnose(trace)
            diag_4 = d4.diagnose(trace)

            natural_failures.append(
                NaturalFailureRecord(
                    question_id=rec.question_id,
                    question=rec.question,
                    gold_answer=rec.gold_answer,
                    generated_answer=trace.answer,
                    retrieved_passage_ids=trace.pre_assembly_passage_ids,
                    context_passage_ids=trace.post_assembly_passage_ids,
                    adjudicated_cause=adjudicated,
                    d1_prediction=diag_1.predicted_cause,
                    d2_prediction=diag_2.predicted_cause,
                    d4_prediction=diag_4.predicted_cause,
                    d4_abstained=diag_4.abstained,
                    d4_rationale=diag_4.evidence_rationale,
                )
            )

    failure_count = len(natural_failures)
    clean_accuracy = (total_evaluated - failure_count) / total_evaluated if total_evaluated > 0 else 0.0

    # Calculate agreement between D4 prediction and adjudicated cause on natural failures
    matches = sum(1 for f in natural_failures if f.d4_prediction == f.adjudicated_cause)
    transfer_accuracy = matches / failure_count if failure_count > 0 else 0.0

    return {
        "total_evaluated_clean_questions": total_evaluated,
        "clean_pipeline_accuracy": round(clean_accuracy, 4),
        "total_natural_failures": failure_count,
        "natural_failure_rate": round(failure_count / total_evaluated, 4) if total_evaluated > 0 else 0.0,
        "adjudicated_cause_breakdown": dict(cause_counts),
        "d4_transfer_agreement_rate": round(transfer_accuracy, 4),
        "failures": [f.model_dump() for f in natural_failures],
    }
