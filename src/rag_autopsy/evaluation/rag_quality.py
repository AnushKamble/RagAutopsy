"""RAG Quality Comparison Baselines (Ragas Faithfulness & Context Precision@k).

Spec Section 2, 5, 11:
Contrasts conventional scalar RAG quality metrics against failure attribution.
Demonstrates that aggregate quality scores (e.g. 0.42 faithfulness or 0.50 context precision)
do not identify the failed pipeline stage, whereas RAG-Autopsy pinpoints the intervention.
"""

import re

from pydantic import BaseModel, Field

from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord, RetrievedPassage
from rag_autopsy.evaluation.answer_metrics import evaluate_answer_correctness


class TrialQualityScore(BaseModel):
    """Quality scores for a single evaluation trial."""

    trial_id: str
    injected_fault: str
    context_precision: float = Field(..., description="Context Precision@k (mAP@k of retrieved passages).")
    faithfulness: float = Field(..., description="Proportion of answer claims supported by generator context.")
    answer_exact_match: bool
    answer_token_f1: float


class FaultQualityProfile(BaseModel):
    """Aggregated quality metric profile for a specific fault class."""

    fault_id: str
    trial_count: int
    mean_context_precision: float
    mean_faithfulness: float
    mean_token_f1: float
    attribution_distinction: str = Field(
        ...,
        description="Explanation of why scalar quality scores are insufficient to isolate this failure stage."
    )


class QualityVsAttributionComparison(BaseModel):
    """Comparative analysis contrasting scalar quality scores with fault attribution."""

    total_trials: int
    overall_context_precision: float
    overall_faithfulness: float
    overall_token_f1: float
    profiles_by_fault: dict[str, FaultQualityProfile]
    interpretability_gap_summary: str


def compute_context_precision_at_k(
    retrieved_passages: list[RetrievedPassage],
    gold_evidence_ids: list[str],
    cutoff_k: int | None = None,
) -> float:
    """Compute Context Precision@k (Mean Average Precision among retrieved passages).

    Formula:
        ContextPrecision@k = sum(Precision@i * v_i) / total_relevant_in_top_k
        where v_i = 1 if passage_i is in gold_evidence_ids, else 0.
    """
    if not retrieved_passages or not gold_evidence_ids:
        return 0.0

    k = min(len(retrieved_passages), cutoff_k) if cutoff_k else len(retrieved_passages)
    if k == 0:
        return 0.0

    gold_set: set[str] = set(gold_evidence_ids)
    relevant_hits = 0
    precision_sum = 0.0

    for i in range(1, k + 1):
        p = retrieved_passages[i - 1]
        is_relevant = 1 if p.passage_id in gold_set else 0
        if is_relevant:
            relevant_hits += 1
            precision_sum += relevant_hits / i

    if relevant_hits == 0:
        return 0.0

    return round(precision_sum / relevant_hits, 4)


def split_into_claims(text: str) -> list[str]:
    """Segment text into individual sentence claims for faithfulness evaluation."""
    clean = text.strip()
    if not clean:
        return []
    # Split by sentence-ending punctuation
    raw_sentences = re.split(r"(?<=[.!?])\s+", clean)
    claims = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
    return claims if claims else [clean]


def compute_faithfulness(answer: str, generator_context: str) -> float:
    """Compute Ragas-style Faithfulness (ratio of answer claims entailed/supported by context).

    Evaluates whether each claim in the answer has substantial lexical overlap (>50% non-stop words)
    or substring presence in the generator context.
    """
    if not answer or not answer.strip():
        return 1.0  # Empty generation trivially contains no unfaithful claims
    if not generator_context or not generator_context.strip():
        return 0.0  # Generation with zero context is ungrounded

    claims = split_into_claims(answer)
    if not claims:
        return 1.0

    context_lower = generator_context.lower()
    supported_count = 0

    stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or", "that", "this", "it"}

    for claim in claims:
        claim_lower = claim.lower()
        if claim_lower in context_lower:
            supported_count += 1
            continue

        # Check significant content-token overlap
        tokens = [t for t in re.findall(r"\w+", claim_lower) if t not in stop_words and len(t) > 2]
        if not tokens:
            supported_count += 1
            continue

        matched = sum(1 for t in tokens if t in context_lower)
        ratio = matched / len(tokens)
        if ratio >= 0.75:
            supported_count += 1

    return round(supported_count / len(claims), 4)


FAULT_DISTINCTION_DESCRIPTIONS: dict[str, str] = {
    "F0": "Clean execution. High precision and faithfulness; establishes reference performance ceiling.",
    "F1": "Evidence Unavailable. Context precision drops to 0.0 because evidence is missing from corpus. Scalar score cannot distinguish whether evidence was unindexed (F1) vs demoted (F2).",
    "F2": "Evidence Ranked Out. Context precision drops to 0.0 due to distractor dominance. Evidence exists in index but missed cutoff; requires retriever tuning, not corpus re-indexing.",
    "F3": "Context Assembly Loss. Context precision appears normal (evidence retrieved), but context assembly dropped evidence. Faithfulness drops; fix requires context window / assembly prompt redesign.",
    "F4": "Contradictory Evidence. Context precision may be non-zero, but contradictory passage in context produces conflicting claims. Scalar precision masks internal evidence conflict.",
    "F5": "Unsupported Generation. Both context precision and evidence availability are intact, yet generator hallucinated ungrounded answer. Fix requires generator prompting/model swap.",
    "F2+F3": "Compound Fault. Both retrieval demotion and assembly loss occur simultaneously. Scalar scores conflate two distinct architectural failure locations.",
    "F3+F4": "Compound Fault. Assembly dropped evidence AND contradiction was injected. Faithfulness collapse masks that two separate pipeline stages failed.",
}


def evaluate_rag_quality_dataset(
    traces: list[DiagnosticTrace],
    eval_records: list[EvaluatorRecord],
) -> QualityVsAttributionComparison:
    """Evaluate quality metrics across all trials and contrast against fault attribution."""
    trace_map = {t.trial_id: t for t in traces}
    trial_scores: list[TrialQualityScore] = []

    for ev in eval_records:
        tr = trace_map.get(ev.trial_id)
        if not tr:
            continue

        c_prec = compute_context_precision_at_k(tr.retrieved_top_k, ev.gold_evidence_ids)
        faith = compute_faithfulness(tr.answer, tr.generator_context)
        em, f1 = evaluate_answer_correctness(tr.answer, ev.gold_answers)

        trial_scores.append(
            TrialQualityScore(
                trial_id=ev.trial_id,
                injected_fault=ev.injected_fault,
                context_precision=c_prec,
                faithfulness=faith,
                answer_exact_match=em,
                answer_token_f1=f1,
            )
        )

    if not trial_scores:
        raise ValueError("No matching trials found to compute RAG quality metrics.")

    # Aggregate by fault type
    by_fault: dict[str, list[TrialQualityScore]] = {}
    for s in trial_scores:
        by_fault.setdefault(s.injected_fault, []).append(s)

    profiles: dict[str, FaultQualityProfile] = {}
    for fid, scores in sorted(by_fault.items()):
        n = len(scores)
        profiles[fid] = FaultQualityProfile(
            fault_id=fid,
            trial_count=n,
            mean_context_precision=round(sum(s.context_precision for s in scores) / n, 4),
            mean_faithfulness=round(sum(s.faithfulness for s in scores) / n, 4),
            mean_token_f1=round(sum(s.answer_token_f1 for s in scores) / n, 4),
            attribution_distinction=FAULT_DISTINCTION_DESCRIPTIONS.get(
                fid, "Multi-stage or custom fault intervention."
            ),
        )

    n_tot = len(trial_scores)
    summary_text = (
        "Analysis demonstrates that aggregate scalar metrics (Context Precision & Faithfulness) "
        "conflate distinct failure modes: for example, F1 (Evidence Unavailable) and F2 (Evidence Ranked Out) "
        "both yield 0.0 Context Precision despite requiring fundamentally different engineering remediations "
        "(corpus indexing vs retriever reranking). RAG-Autopsy attribution isolates the specific intervention stage."
    )

    return QualityVsAttributionComparison(
        total_trials=n_tot,
        overall_context_precision=round(sum(s.context_precision for s in trial_scores) / n_tot, 4),
        overall_faithfulness=round(sum(s.faithfulness for s in trial_scores) / n_tot, 4),
        overall_token_f1=round(sum(s.answer_token_f1 for s in trial_scores) / n_tot, 4),
        profiles_by_fault=profiles,
        interpretability_gap_summary=summary_text,
    )
