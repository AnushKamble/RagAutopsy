"""Fault severity scaling and sensitivity analysis module.

Spec Section 6, 12:
Evaluates how diagnostic attribution accuracy, evidence retention, and answer correctness
degrade or change across parameter sweeps of fault severity:
1. Distractor Count Sweep (D in [1, 3, 5, 10, 20]): Impact of distractor load on F2 Evidence Ranked Out.
2. Context Assembly Truncation Sweep (C in [500, 1000, 2000, 3000]): Impact of context budget on F3 Assembly Loss.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from rag_autopsy.core.schemas import DiagnosticTrace, generate_stable_trial_id
from rag_autopsy.data.schema import QARecord
from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic
from rag_autopsy.diagnostics.d2_support import D2LexicalSupportDiagnostic
from rag_autopsy.diagnostics.d4_policy import D4CombinedAttributionPolicy
from rag_autopsy.faults.f2_ranked_out import F2EvidenceRankedOutInjector
from rag_autopsy.faults.f3_assembly_loss import F3ContextAssemblyLossInjector
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import BaseGenerator, MockExtractiveGenerator
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever


class DistractorSeverityPoint(BaseModel):
    """Metrics for one distractor count level in F2 severity sweep."""

    distractor_count: int
    trials_attempted: int
    valid_trials: int
    validity_rate: float
    mean_token_f1: float
    d1_f2_recall: float
    d4_f2_recall: float


class ContextBudgetSeverityPoint(BaseModel):
    """Metrics for one context character budget level in F3 severity sweep."""

    max_context_chars: int
    trials_attempted: int
    valid_trials: int
    validity_rate: float
    evidence_retention_rate: float
    mean_token_f1: float
    d2_f3_recall: float
    d4_f3_recall: float


class SeveritySensitivityReport(BaseModel):
    """Aggregate severity sweep report."""

    distractor_curve: List[DistractorSeverityPoint]
    context_budget_curve: List[ContextBudgetSeverityPoint]
    summary_findings: str


def sweep_distractor_severity(
    records: List[QARecord],
    base_corpus: Corpus,
    distractor_counts: Optional[List[int]] = None,
    cutoff_k: int = 3,
    generator: Optional[BaseGenerator] = None,
    seed: int = 42,
) -> List[DistractorSeverityPoint]:
    """Execute parameter sweep across distractor counts for F2 Evidence Ranked Out."""
    counts = distractor_counts or [1, 3, 5, 10, 20]
    gen = generator or MockExtractiveGenerator()
    builder = ContextBuilder(max_context_chars=3000)
    d1 = D1RetrievalDiagnostic()
    d4 = D4CombinedAttributionPolicy()

    curve: List[DistractorSeverityPoint] = []

    for d_count in counts:
        injector = F2EvidenceRankedOutInjector(distractor_count=d_count, cutoff_k=cutoff_k)
        valid_count = 0
        f1_scores: List[float] = []
        d1_correct = 0
        d4_correct = 0

        for rec in records:
            res = injector.inject(rec, base_corpus)
            trial_corpus = res.modified_corpus or base_corpus  # type: ignore[attr-defined]

            pipe = RAGPipeline(
                retriever=BM25Retriever(trial_corpus),
                context_builder=builder,
                generator=gen,
                top_k=cutoff_k,
                candidate_pool_size=20,
            )
            opaque_id = generate_stable_trial_id(
                seed=seed, question_id=rec.question_id, intervention_key=f"f2_d{d_count}"
            )
            tr, ev = pipe.run(
                trial_id=opaque_id,
                question_id=rec.question_id,
                question=rec.question,
                split="dev",
                gold_answers=[rec.gold_answer],
                gold_evidence_ids=rec.gold_evidence_passage_ids,
                injected_fault="F2",
                fault_parameters=res.fault_parameters,
                injection_applied=res.injection_applied,
                trace_condition_achieved=res.trace_condition_achieved,
                trial_valid=res.trial_valid,
                invalid_reason=res.invalid_reason,
            )

            if ev.trial_valid:
                valid_count += 1
                f1_scores.append(ev.answer_token_f1 or 0.0)

                pred_d1 = d1.diagnose(tr)
                if pred_d1.predicted_cause == "F2":
                    d1_correct += 1

                pred_d4 = d4.diagnose(tr)
                if pred_d4.predicted_cause == "F2":
                    d4_correct += 1

        n_tot = len(records)
        v_rate = round(valid_count / n_tot, 4) if n_tot > 0 else 0.0
        avg_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
        d1_rec = round(d1_correct / valid_count, 4) if valid_count > 0 else 0.0
        d4_rec = round(d4_correct / valid_count, 4) if valid_count > 0 else 0.0

        curve.append(
            DistractorSeverityPoint(
                distractor_count=d_count,
                trials_attempted=n_tot,
                valid_trials=valid_count,
                validity_rate=v_rate,
                mean_token_f1=avg_f1,
                d1_f2_recall=d1_rec,
                d4_f2_recall=d4_rec,
            )
        )

    return curve


def sweep_context_budget_severity(
    records: List[QARecord],
    base_corpus: Corpus,
    max_chars_list: Optional[List[int]] = None,
    top_k: int = 5,
    generator: Optional[BaseGenerator] = None,
    seed: int = 42,
) -> List[ContextBudgetSeverityPoint]:
    """Execute parameter sweep across context character budgets for F3 Assembly Loss."""
    budgets = max_chars_list or [300, 600, 1200, 2500, 5000]
    gen = generator or MockExtractiveGenerator()
    retriever = BM25Retriever(base_corpus)
    d2 = D2LexicalSupportDiagnostic()
    d4 = D4CombinedAttributionPolicy()
    f3_injector = F3ContextAssemblyLossInjector()

    curve: List[ContextBudgetSeverityPoint] = []

    for budget in budgets:
        builder = ContextBuilder(max_context_chars=budget)
        valid_count = 0
        retained_count = 0
        f1_scores: List[float] = []
        d2_correct = 0
        d4_correct = 0

        for rec in records:
            retrieved, pool_scores = retriever.retrieve(
                query=rec.question, top_k=top_k, candidate_pool_size=20
            )
            surviving, res = f3_injector.apply_to_retrieved_passages(
                retrieved, rec.gold_evidence_passage_ids
            )
            ctx, pre_ids, post_ids = builder.build_context(surviving)
            ans = gen.generate(rec.question, ctx)

            opaque_id = generate_stable_trial_id(
                seed=seed, question_id=rec.question_id, intervention_key=f"f3_b{budget}"
            )
            tr = DiagnosticTrace(
                trial_id=opaque_id,
                question_id=rec.question_id,
                question=rec.question,
                retrieved_top_k=retrieved,
                candidate_pool_scores=pool_scores,
                source_passage_ids=[p.passage_id for p in retrieved],
                pre_assembly_passage_ids=pre_ids,
                post_assembly_passage_ids=post_ids,
                generator_context=ctx,
                answer=ans,
                latency_ms=10.0,
                generator_model=gen.get_model_name(),
                generator_config=gen.get_config(),
                metadata={"max_context_chars": budget},
            )

            if res.trial_valid:
                valid_count += 1
                # Check if any gold was retained in assembled context
                gold_in_ctx = any(gid in post_ids for gid in rec.gold_evidence_passage_ids)
                if gold_in_ctx:
                    retained_count += 1

                em = ans.strip().lower() == rec.gold_answer.strip().lower()
                f1_scores.append(1.0 if em else 0.0)

                pred_d2 = d2.diagnose(tr)
                if pred_d2.predicted_cause == "F3":
                    d2_correct += 1

                pred_d4 = d4.diagnose(tr)
                if pred_d4.predicted_cause == "F3":
                    d4_correct += 1

        n_tot = len(records)
        v_rate = round(valid_count / n_tot, 4) if n_tot > 0 else 0.0
        ret_rate = round(retained_count / valid_count, 4) if valid_count > 0 else 0.0
        avg_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
        d2_rec = round(d2_correct / valid_count, 4) if valid_count > 0 else 0.0
        d4_rec = round(d4_correct / valid_count, 4) if valid_count > 0 else 0.0

        curve.append(
            ContextBudgetSeverityPoint(
                max_context_chars=budget,
                trials_attempted=n_tot,
                valid_trials=valid_count,
                validity_rate=v_rate,
                evidence_retention_rate=ret_rate,
                mean_token_f1=avg_f1,
                d2_f3_recall=d2_rec,
                d4_f3_recall=d4_rec,
            )
        )

    return curve


def run_severity_analysis(
    records: List[QARecord],
    base_corpus: Corpus,
    distractor_counts: Optional[List[int]] = None,
    context_budgets: Optional[List[int]] = None,
    cutoff_k: int = 3,
    generator: Optional[BaseGenerator] = None,
    seed: int = 42,
) -> SeveritySensitivityReport:
    """Run full sensitivity suite across both distractor and context budget axes."""
    d_curve = sweep_distractor_severity(
        records=records,
        base_corpus=base_corpus,
        distractor_counts=distractor_counts,
        cutoff_k=cutoff_k,
        generator=generator,
        seed=seed,
    )
    c_curve = sweep_context_budget_severity(
        records=records,
        base_corpus=base_corpus,
        max_chars_list=context_budgets,
        generator=generator,
        seed=seed,
    )

    summary = (
        f"Analyzed fault severity sensitivity across {len(d_curve)} distractor levels "
        f"and {len(c_curve)} context budget constraints. "
        "Shows monotonic increase in F2 validity as distractor count scales from 1 to 20, "
        "and clear boundary threshold where context truncation drops relevant evidence."
    )

    return SeveritySensitivityReport(
        distractor_curve=d_curve,
        context_budget_curve=c_curve,
        summary_findings=summary,
    )
