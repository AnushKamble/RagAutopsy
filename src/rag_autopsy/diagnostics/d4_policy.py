"""D4: Combined attribution policy engine with abstention logic."""

import time

from rag_autopsy.core.schemas import Diagnosis, DiagnosticTrace
from rag_autopsy.diagnostics.base import BaseDiagnostic
from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic
from rag_autopsy.diagnostics.d2_support import D2LexicalSupportDiagnostic


class D4CombinedAttributionPolicy(BaseDiagnostic):
    """Multi-signal policy synthesizing retrieval and lexical diagnostics."""

    def __init__(
        self,
        name: str = "d4_combined_policy",
        regime: str = "regime_a",
        abstention_enabled: bool = True,
        confidence_threshold: float = 0.65,
        low_score_threshold: float = 2.0,
        score_gap_threshold: float = 1.0,
        min_overlap_ratio: float = 0.3,
        enable_nli: bool = False,
    ):
        super().__init__(name=name)
        self.regime = regime
        self.abstention_enabled = abstention_enabled
        self.confidence_threshold = confidence_threshold
        self.d1 = D1RetrievalDiagnostic(
            regime=regime,
            low_score_threshold=low_score_threshold,
            score_gap_threshold=score_gap_threshold,
            abstain_on_ambiguity=abstention_enabled,
        )
        self.d2 = D2LexicalSupportDiagnostic(min_overlap_ratio=min_overlap_ratio)
        if enable_nli:
            from rag_autopsy.diagnostics.d3_nli import D3NLISupportDiagnostic
            self.d3 = D3NLISupportDiagnostic()
        else:
            self.d3 = None

    def _predict(self, trace: DiagnosticTrace) -> Diagnosis:
        start = time.perf_counter()

        diag_d1 = self.d1.diagnose(trace)
        diag_d2 = self.d2.diagnose(trace)

        # Rule 1: Assembly Loss takes precedence if structural omission is evident
        if diag_d1.predicted_cause == "F3" or diag_d2.predicted_cause == "F3":
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause="F3",
                confidence=0.95,
                abstained=False,
                evidence_rationale="Structural omission: passages retrieved in top-k were dropped during context assembly.",
                runtime_ms=round(elapsed_ms, 3),
            )

        # Rule 2: Contradictory Evidence takes precedence if detected in context
        if diag_d2.predicted_cause == "F4":
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause="F4",
                confidence=0.88,
                abstained=False,
                evidence_rationale="Material contradiction cues identified across generator context passages.",
                runtime_ms=round(elapsed_ms, 3),
            )

        # Rule 3: Retrieval failures
        if diag_d1.predicted_cause in ["F1", "F2", "F1_F2_RETRIEVAL_MISS"]:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause=diag_d1.predicted_cause,
                confidence=diag_d1.confidence,
                abstained=diag_d1.abstained,
                abstention_reason=diag_d1.abstention_reason,
                evidence_rationale=f"Combined policy affirms retrieval diagnosis: {diag_d1.evidence_rationale}",
                runtime_ms=round(elapsed_ms, 3),
            )

        # Rule 4: Check for F5 Unsupported Generation if NLI model is enabled
        if self.d3:
            diag_d3 = self.d3.diagnose(trace)
            if diag_d3.predicted_cause == "F5" and diag_d1.predicted_cause == "F0":
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return Diagnosis(
                    trial_id=trace.trial_id,
                    diagnostic_name=self.name,
                    predicted_cause="F5",
                    confidence=diag_d3.confidence,
                    abstained=False,
                    evidence_rationale=f"Combined policy affirms F5 unsupported generation: {diag_d3.evidence_rationale}",
                    runtime_ms=round(elapsed_ms, 3),
                )

        # Rule 5: Grounding confirmation
        if diag_d1.predicted_cause == "F0" and diag_d2.predicted_cause == "F0":
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause="F0",
                confidence=0.85,
                abstained=False,
                evidence_rationale="Both retrieval scores and lexical context grounding indicate clean execution.",
                runtime_ms=round(elapsed_ms, 3),
            )

        # Rule 5: Fallback / Abstention on low confidence or conflicting signals
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if self.abstention_enabled:
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause="UNKNOWN",
                confidence=0.4,
                abstained=True,
                abstention_reason="Conflicting signals between retrieval scores and lexical grounding.",
                evidence_rationale=f"D1 predicted {diag_d1.predicted_cause}, D2 predicted {diag_d2.predicted_cause}.",
                runtime_ms=round(elapsed_ms, 3),
            )

        return Diagnosis(
            trial_id=trace.trial_id,
            diagnostic_name=self.name,
            predicted_cause=diag_d1.predicted_cause,
            confidence=0.5,
            abstained=False,
            evidence_rationale=f"Defaulted to D1 signal: {diag_d1.predicted_cause}",
            runtime_ms=round(elapsed_ms, 3),
        )
