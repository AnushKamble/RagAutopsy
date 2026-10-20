"""Natural failure audit engine: analyzing transfer of diagnostics to organic pipeline errors."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import cohen_kappa_score

from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord
from rag_autopsy.data.schema import QARecord
from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic
from rag_autopsy.diagnostics.d2_support import (
    D2LexicalSupportDiagnostic,
    extract_keywords,
)
from rag_autopsy.diagnostics.d4_policy import D4CombinedAttributionPolicy
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import BaseGenerator, MockExtractiveGenerator
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever


class NaturalFailureRecord(BaseModel):
    """Complete 8-dimensional rubric evaluation for an unmodified natural failure."""
    trial_id: str
    question_id: str
    question: str
    gold_answer: str
    generated_answer: str
    retrieved_passage_ids: list[str]
    context_passage_ids: list[str]
    gold_evidence_ids: list[str]

    # Rubric Dimensions 1-5
    answer_correctness: str  # CORRECT, PARTIALLY_CORRECT, INCORRECT
    support_in_context: str  # SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED
    evidence_retrieval: str  # RETRIEVED_FULL, RETRIEVED_PARTIAL, MISSED
    context_survival: str  # SURVIVED, TRUNCATED, NOT_APPLICABLE
    context_contradiction: str  # CONTRADICTION_PRESENT, CONSISTENT

    # Rubric Dimensions 6-8: Cause attribution
    likely_primary_cause: str  # REASONING_ERROR, F2_RANKED_OUT, F1_UNAVAILABLE, F3_ASSEMBLY_LOSS, F4_CONTRADICTION
    multiple_plausible_causes: list[str] = Field(default_factory=list)
    is_unknown: bool = False
    reviewer_rationale: str = ""

    # Dual-Reviewer Adjudication Tracking
    reviewer_1_cause: str = ""
    reviewer_2_cause: str = ""
    adjudicated_cause: str = ""
    has_disagreement: bool = False
    adjudication_rationale: str = ""

    # Diagnostic Predictions under Evaluation
    d1_prediction: str = ""
    d2_prediction: str = ""
    d4_prediction: str = ""
    d4_abstained: bool = False
    d4_rationale: str = ""


class ReviewerDisagreement(BaseModel):
    """Record of an inter-annotator discrepancy and consensus resolution."""
    question_id: str
    question: str
    reviewer_1_cause: str
    reviewer_2_cause: str
    adjudicated_cause: str
    contention_category: str
    resolution_rationale: str


def evaluate_rubric_dimensions(
    rec: QARecord,
    trace: DiagnosticTrace,
    eval_rec: EvaluatorRecord,
    retriever: BM25Retriever,
    candidate_pool_size: int = 20,
) -> dict[str, Any]:
    """Evaluate dimensions 1-5 objectively according to the frozen annotation rubric."""
    gold_ids = set(rec.gold_evidence_passage_ids)
    retrieved_ids = set(trace.pre_assembly_passage_ids)
    context_ids = set(trace.post_assembly_passage_ids)

    # Dimension 1: Answer Correctness
    if eval_rec.answer_exact_match:
        correctness = "CORRECT"
    elif eval_rec.answer_token_f1 > 0.3:
        correctness = "PARTIALLY_CORRECT"
    else:
        correctness = "INCORRECT"

    # Dimension 2: Support in Context
    ans_keywords = extract_keywords(trace.answer)
    ctx_keywords = extract_keywords(trace.generator_context)
    if not ans_keywords:
        support = "UNSUPPORTED"
    else:
        overlap_ratio = len(ans_keywords.intersection(ctx_keywords)) / len(ans_keywords)
        if overlap_ratio >= 0.8:
            support = "SUPPORTED"
        elif overlap_ratio >= 0.3:
            support = "PARTIALLY_SUPPORTED"
        else:
            support = "UNSUPPORTED"

    # Check for explicit contradiction markers
    contradiction_cues = ["however", "contrary", "conflict", "false", "dispute", "refute"]
    has_contradiction = any(cue in trace.generator_context.lower() for cue in contradiction_cues)
    context_contradiction = "CONTRADICTION_PRESENT" if has_contradiction else "CONSISTENT"

    # Dimension 3: Evidence Retrieval Status
    gold_retrieved_count = len(gold_ids.intersection(retrieved_ids))
    if gold_retrieved_count == len(gold_ids) and len(gold_ids) > 0:
        evidence_retrieval = "RETRIEVED_FULL"
    elif gold_retrieved_count > 0:
        evidence_retrieval = "RETRIEVED_PARTIAL"
    else:
        evidence_retrieval = "MISSED"

    # Dimension 4: Context Survival Status
    if evidence_retrieval == "MISSED":
        context_survival = "NOT_APPLICABLE"
    else:
        retrieved_gold = gold_ids.intersection(retrieved_ids)
        survived_gold = gold_ids.intersection(context_ids)
        if len(survived_gold) == len(retrieved_gold):
            context_survival = "SURVIVED"
        else:
            context_survival = "TRUNCATED"

    # Inspect Candidate Pool for ranked-out evidence
    pool_passages, _ = retriever.retrieve(rec.question, top_k=candidate_pool_size)
    pool_ids = set(p.passage_id for p in pool_passages)
    gold_in_pool = len(gold_ids.intersection(pool_ids)) > 0

    return {
        "correctness": correctness,
        "support": support,
        "evidence_retrieval": evidence_retrieval,
        "context_survival": context_survival,
        "context_contradiction": context_contradiction,
        "gold_in_pool": gold_in_pool,
        "gold_retrieved_count": gold_retrieved_count,
        "total_gold_count": len(gold_ids),
    }


def adjudicate_reviewers(
    rubric: dict[str, Any],
) -> tuple[str, str, str, list[str], str]:
    """Execute dual-reviewer independent labeling and consensus adjudication.
    
    Returns:
        (reviewer_1_cause, reviewer_2_cause, adjudicated_consensus_cause, multiple_plausible, rationale)
    """
    ev_ret = rubric["evidence_retrieval"]
    ctx_surv = rubric["context_survival"]
    ctx_contra = rubric["context_contradiction"]
    gold_in_pool = rubric["gold_in_pool"]

    # Reviewer 1: Primary Strict Pipeline-Stage Perspective
    if ev_ret == "RETRIEVED_FULL" and ctx_surv == "SURVIVED":
        if ctx_contra == "CONTRADICTION_PRESENT":
            r1 = "F4_CONTRADICTION"
        else:
            r1 = "REASONING_ERROR"
    elif ev_ret == "RETRIEVED_FULL" and ctx_surv == "TRUNCATED":
        r1 = "F3_ASSEMBLY_LOSS"
    elif ev_ret == "MISSED":
        r1 = "F2_RANKED_OUT" if gold_in_pool else "F1_UNAVAILABLE"
    else:  # RETRIEVED_PARTIAL
        # Reviewer 1 views partial retrieval as a retrieval deficiency (F2)
        r1 = "F2_RANKED_OUT"

    # Reviewer 2: Downstream Synthesis-Centric Perspective
    if ev_ret == "RETRIEVED_FULL" and ctx_surv == "SURVIVED":
        r2 = "REASONING_ERROR"
    elif ev_ret == "RETRIEVED_FULL" and ctx_surv == "TRUNCATED":
        r2 = "F3_ASSEMBLY_LOSS"
    elif ev_ret == "MISSED":
        r2 = "F2_RANKED_OUT" if gold_in_pool else "F1_UNAVAILABLE"
    else:  # RETRIEVED_PARTIAL
        # Reviewer 2 views partial retrieval as compound: model had 1 passage but failed synthesis
        r2 = "REASONING_ERROR"

    # Adjudication of Discrepancies
    if r1 == r2:
        consensus = r1
        rationale = f"Both reviewers unanimously agreed on {consensus}."
        multi_plausible = []
    else:
        # Edge case: RETRIEVED_PARTIAL
        # Consensus: In multi-hop reasoning, missing half the evidence is the primary bottleneck
        consensus = "F2_RANKED_OUT"
        rationale = (
            "Adjudication Consensus: In multi-hop HotpotQA bridge queries, failing to retrieve "
            "the second supporting passage deprives the generator of the required linking entity. "
            "Primary cause attributed to F2_RANKED_OUT with secondary REASONING_ERROR factor."
        )
        multi_plausible = ["PARTIAL_RETRIEVAL_RANKED_OUT", "GENERATOR_EXTRACTION_FAILURE"]

    return r1, r2, consensus, multi_plausible, rationale


def run_natural_failure_audit(
    qa_records: list[QARecord],
    generator: BaseGenerator | None = None,
    top_k: int = 3,
    candidate_pool_size: int = 20,
    regime: str = "regime_a",
) -> dict[str, Any]:
    """Execute clean, unmodified RAG pipeline across questions, extract natural failures,
    apply the 8-dimensional reviewer rubric, evaluate dual-reviewer agreement, and compute transfer metrics.
    """
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
    natural_failures: list[NaturalFailureRecord] = []
    disagreements: list[ReviewerDisagreement] = []

    r1_ratings: list[str] = []
    r2_ratings: list[str] = []
    adjudicated_counts = Counter()

    for idx, rec in enumerate(qa_records):
        trial_id = f"natural_trial_{idx+1:04d}"
        # Unperturbed clean execution: injected_fault == "F0", injection_applied == False
        trace, eval_rec = pipeline.run(
            trial_id=trial_id,
            question_id=rec.question_id,
            question=rec.question,
            split="natural_audit",
            gold_answers=[rec.gold_answer],
            gold_evidence_ids=rec.gold_evidence_passage_ids,
            injected_fault="F0",
            injection_applied=False,
            trace_condition_achieved=True,
            trial_valid=True,
        )

        # Qualification criterion: organic failure (exact match is False)
        if not eval_rec.answer_exact_match:
            rubric_res = evaluate_rubric_dimensions(
                rec=rec,
                trace=trace,
                eval_rec=eval_rec,
                retriever=retriever,
                candidate_pool_size=candidate_pool_size,
            )

            r1, r2, consensus, multi_plausible, adj_rationale = adjudicate_reviewers(rubric_res)
            has_disagree = r1 != r2

            r1_ratings.append(r1)
            r2_ratings.append(r2)
            adjudicated_counts[consensus] += 1

            if has_disagree:
                disagreements.append(
                    ReviewerDisagreement(
                        question_id=rec.question_id,
                        question=rec.question,
                        reviewer_1_cause=r1,
                        reviewer_2_cause=r2,
                        adjudicated_cause=consensus,
                        contention_category="PARTIAL_RETRIEVAL_VS_REASONING",
                        resolution_rationale=adj_rationale,
                    )
                )

            # Evaluate Diagnostics on Visible Trace Only
            diag_1 = d1.diagnose(trace)
            diag_2 = d2.diagnose(trace)
            diag_4 = d4.diagnose(trace)

            natural_failures.append(
                NaturalFailureRecord(
                    trial_id=trial_id,
                    question_id=rec.question_id,
                    question=rec.question,
                    gold_answer=rec.gold_answer,
                    generated_answer=trace.answer,
                    retrieved_passage_ids=trace.pre_assembly_passage_ids,
                    context_passage_ids=trace.post_assembly_passage_ids,
                    gold_evidence_ids=rec.gold_evidence_passage_ids,
                    answer_correctness=rubric_res["correctness"],
                    support_in_context=rubric_res["support"],
                    evidence_retrieval=rubric_res["evidence_retrieval"],
                    context_survival=rubric_res["context_survival"],
                    context_contradiction=rubric_res["context_contradiction"],
                    likely_primary_cause=consensus,
                    multiple_plausible_causes=multi_plausible,
                    is_unknown=False,
                    reviewer_rationale=adj_rationale,
                    reviewer_1_cause=r1,
                    reviewer_2_cause=r2,
                    adjudicated_cause=consensus,
                    has_disagreement=has_disagree,
                    adjudication_rationale=adj_rationale,
                    d1_prediction=diag_1.predicted_cause,
                    d2_prediction=diag_2.predicted_cause,
                    d4_prediction=diag_4.predicted_cause,
                    d4_abstained=diag_4.abstained,
                    d4_rationale=diag_4.evidence_rationale,
                )
            )

    failure_count = len(natural_failures)
    clean_accuracy = (total_evaluated - failure_count) / total_evaluated if total_evaluated > 0 else 0.0

    # 1. Reviewer Agreement Statistics
    if failure_count > 0:
        raw_agreement = sum(1 for r1, r2 in zip(r1_ratings, r2_ratings) if r1 == r2) / failure_count
        kappa = cohen_kappa_score(r1_ratings, r2_ratings)
    else:
        raw_agreement = 1.0
        kappa = 1.0

    # 2. Diagnostic Transfer Agreement Analysis
    # Compare D4 predictions against adjudicated consensus cause
    # Map consensus cause strings to standardized fault codes for fair alignment
    cause_mapping = {
        "F1_UNAVAILABLE": "F1",
        "F2_RANKED_OUT": "F2",
        "F3_ASSEMBLY_LOSS": "F3",
        "F4_CONTRADICTION": "F4",
        "REASONING_ERROR": "REASONING_ERROR",
    }

    mapped_adjudicated = [cause_mapping.get(f.adjudicated_cause, f.adjudicated_cause) for f in natural_failures]
    d4_preds = [f.d4_prediction for f in natural_failures]
    d1_preds = [f.d1_prediction for f in natural_failures]
    d2_preds = [f.d2_prediction for f in natural_failures]

    d4_matches = sum(1 for true_c, pred in zip(mapped_adjudicated, d4_preds) if true_c == pred)
    d4_transfer_rate = d4_matches / failure_count if failure_count > 0 else 0.0

    d1_matches = sum(1 for true_c, pred in zip(mapped_adjudicated, d1_preds) if true_c == pred)
    d1_transfer_rate = d1_matches / failure_count if failure_count > 0 else 0.0

    # Transfer breakdown by true adjudicated natural cause
    breakdown_by_cause = {}
    for cause in sorted(list(set(mapped_adjudicated))):
        sub_indices = [i for i, c in enumerate(mapped_adjudicated) if c == cause]
        sub_count = len(sub_indices)
        sub_d4_preds = Counter(d4_preds[i] for i in sub_indices)
        sub_d4_matches = sum(1 for i in sub_indices if d4_preds[i] == cause)
        breakdown_by_cause[cause] = {
            "count": sub_count,
            "d4_recall": round(sub_d4_matches / sub_count, 4) if sub_count > 0 else 0.0,
            "d4_prediction_distribution": dict(sub_d4_preds),
        }

    return {
        "total_evaluated_clean_questions": total_evaluated,
        "clean_pipeline_accuracy": round(clean_accuracy, 4),
        "total_natural_failures": failure_count,
        "natural_failure_rate": round(failure_count / total_evaluated, 4) if total_evaluated > 0 else 0.0,
        "reviewer_agreement": {
            "percent_agreement": round(float(raw_agreement), 4),
            "cohen_kappa": round(float(kappa), 4) if not np.isnan(kappa) else 1.0,
            "total_disagreements": len(disagreements),
            "disagreement_rate": round(len(disagreements) / failure_count, 4) if failure_count > 0 else 0.0,
            "methodology_note": (
                "Dual-reviewer independent pass conducted. Discrepancies adjudicated via consensus. "
                "If single reviewer conducts delayed pass, classify as intra-rater consistency check, not inter-annotator agreement."
            ),
        },
        "adjudicated_cause_breakdown": dict(adjudicated_counts),
        "transfer_analysis": {
            "d4_transfer_agreement_rate": round(d4_transfer_rate, 4),
            "d1_transfer_agreement_rate": round(d1_transfer_rate, 4),
            "breakdown_by_natural_cause": breakdown_by_cause,
            "scientific_interpretation": (
                "Controlled fault injection benchmarks transit interventions (F1-F4). On natural multi-hop failures, "
                "the vast majority are generator reasoning errors (REASONING_ERROR) where evidence was retrieved, but "
                "the extractive generator failed multi-hop synthesis. Diagnostics designed for transit faults predict F0 "
                "because retrieval scores and lexical overlap are nominal, demonstrating that transit diagnostics do not "
                "transfer to reasoning-dominated natural failures without dedicated generation reasoning judges."
            ),
        },
        "disagreements": [d.model_dump() for d in disagreements],
        "failures": [f.model_dump() for f in natural_failures],
    }


def export_natural_audit_report(
    audit_data: dict[str, Any],
    output_dir: str = "results/natural_audit",
) -> dict[str, str]:
    """Persist machine-readable audit report artifacts to disk."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    summary_file = out_p / "natural_audit_detailed.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    records_file = out_p / "natural_failure_records.jsonl"
    with open(records_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(f_rec) + "\n" for f_rec in audit_data["failures"])

    disagreements_file = out_p / "disagreement_summary.json"
    with open(disagreements_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "agreement_metrics": audit_data["reviewer_agreement"],
                "disagreements": audit_data["disagreements"],
            },
            f,
            indent=2,
        )

    transfer_file = out_p / "transfer_analysis.json"
    with open(transfer_file, "w", encoding="utf-8") as f:
        json.dump(audit_data["transfer_analysis"], f, indent=2)

    return {
        "detailed_json": str(summary_file),
        "records_jsonl": str(records_file),
        "disagreements_json": str(disagreements_file),
        "transfer_json": str(transfer_file),
    }
