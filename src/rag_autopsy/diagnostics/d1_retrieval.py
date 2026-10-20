"""D1: Retrieval and ranking diagnostic heuristics."""

import time

from rag_autopsy.core.schemas import Diagnosis, DiagnosticTrace
from rag_autopsy.diagnostics.base import BaseDiagnostic


class D1RetrievalDiagnostic(BaseDiagnostic):
    """Diagnoses failures using retrieval score distributions, candidate pool, and pre/post assembly diffs."""

    def __init__(
        self,
        name: str = "d1_retrieval_rules",
        regime: str = "regime_a",
        low_score_threshold: float = 2.0,
        score_gap_threshold: float = 1.0,
        abstain_on_ambiguity: bool = True,
    ):
        super().__init__(name=name)
        self.regime = regime  # "regime_a" (candidate pool probe) | "regime_b" (blind top-k)
        self.low_score_threshold = low_score_threshold
        self.score_gap_threshold = score_gap_threshold
        self.abstain_on_ambiguity = abstain_on_ambiguity

    def _predict(self, trace: DiagnosticTrace) -> Diagnosis:
        start = time.perf_counter()

        # Signal 1: Context Assembly Loss (Structural Deterministic Diff)
        pre_ids = set(trace.pre_assembly_passage_ids)
        post_ids = set(trace.post_assembly_passage_ids)
        dropped_ids = pre_ids - post_ids

        if dropped_ids:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause="F3",
                confidence=0.95,
                abstained=False,
                evidence_rationale=(
                    f"Structural context loss detected: {len(dropped_ids)} retrieved passage(s) "
                    f"omitted before generator context ({list(dropped_ids)[:2]})."
                ),
                runtime_ms=round(elapsed_ms, 3),
            )

        # Signal 2: Top-k scores
        scores = [p.score for p in trace.retrieved_top_k]
        top_score = scores[0] if scores else 0.0

        # Signal 3: Candidate pool scores (if available under Regime A)
        pool_scores = trace.candidate_pool_scores or scores

        if self.regime == "regime_a":
            # In Regime A, we inspect the candidate pool score drop-off
            if top_score < self.low_score_threshold:
                # Evidence completely absent from index -> low matching score across corpus
                pred = "F1"
                conf = 0.80
                rationale = f"Severe retrieval score depression (top score={top_score} < {self.low_score_threshold})."
            elif len(pool_scores) >= 10 and (pool_scores[0] - pool_scores[min(4, len(pool_scores)-1)]) < self.score_gap_threshold:
                # Distractor crowd with tight score margins -> Evidence ranked out into pool
                pred = "F2"
                conf = 0.75
                rationale = (
                    f"Crowded candidate pool with flat score gap "
                    f"({round(pool_scores[0] - pool_scores[min(4, len(pool_scores)-1)], 2)} < {self.score_gap_threshold})."
                )
            elif not trace.candidate_pool_scores and self.abstain_on_ambiguity:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return Diagnosis(
                    trial_id=trace.trial_id,
                    diagnostic_name=self.name,
                    predicted_cause="UNKNOWN",
                    confidence=0.3,
                    abstained=True,
                    abstention_reason="AMBIGUOUS_POOL_DATA: Candidate pool scores missing under Regime A.",
                    evidence_rationale="Missing candidate pool scores in trace; cannot verify rank margin.",
                    runtime_ms=round(elapsed_ms, 3),
                )
            else:
                pred = "F0"
                conf = 0.70
                rationale = f"Strong top retrieval score ({top_score}) with healthy score gap."

        else:
            # Regime B (Strict Top-k Blind Diagnoser)
            # Cannot separate F1 from F2 from top-k alone
            if top_score < self.low_score_threshold:
                if self.abstain_on_ambiguity:
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    return Diagnosis(
                        trial_id=trace.trial_id,
                        diagnostic_name=self.name,
                        predicted_cause="UNKNOWN",
                        confidence=0.3,
                        abstained=True,
                        abstention_reason="AMBIGUOUS_RETRIEVAL_MISS: Cannot separate F1 (absent) from F2 (ranked out) under top-k only trace.",
                        evidence_rationale=f"Depressed top score ({top_score}), unidentifiable between F1 and F2.",
                        runtime_ms=round(elapsed_ms, 3),
                    )
                else:
                    pred = "F1_F2_RETRIEVAL_MISS"
                    conf = 0.80
                    rationale = f"Depressed top retrieval score ({top_score}), indicating retrieval omission."
            else:
                pred = "F0"
                conf = 0.70
                rationale = f"Healthy top retrieval score ({top_score})."

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return Diagnosis(
            trial_id=trace.trial_id,
            diagnostic_name=self.name,
            predicted_cause=pred,
            confidence=conf,
            abstained=False,
            evidence_rationale=rationale,
            runtime_ms=round(elapsed_ms, 3),
        )
