"""Simple, interpretable baselines: Majority-Cause and Random."""

from collections import Counter
import random
import time
from typing import List, Optional

from rag_autopsy.core.schemas import Diagnosis, DiagnosticTrace
from rag_autopsy.diagnostics.base import BaseDiagnostic


class MajorityCauseBaseline(BaseDiagnostic):
    """Predicts the most frequent cause observed in the calibration/training set."""

    def __init__(self, majority_cause: str = "F0"):
        super().__init__(name="baseline_majority_cause")
        self.majority_cause = majority_cause

    def fit(self, training_causes: List[str]) -> None:
        """Fit the baseline on calibration/training labels."""
        if not training_causes:
            return
        counts = Counter(training_causes)
        self.majority_cause = counts.most_common(1)[0][0]

    def _predict(self, trace: DiagnosticTrace) -> Diagnosis:
        start = time.perf_counter()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return Diagnosis(
            trial_id=trace.trial_id,
            diagnostic_name=self.name,
            predicted_cause=self.majority_cause,
            confidence=0.5,
            abstained=False,
            evidence_rationale=f"Majority-cause baseline prediction: {self.majority_cause}",
            runtime_ms=round(elapsed_ms, 3),
        )


class RandomBaseline(BaseDiagnostic):
    """Predicts a cause uniformly at random among candidate classes."""

    def __init__(self, candidate_causes: Optional[List[str]] = None, seed: int = 42):
        super().__init__(name="baseline_random")
        self.candidate_causes = candidate_causes or ["F0", "F1", "F2", "F3", "F4"]
        self.rng = random.Random(seed)

    def _predict(self, trace: DiagnosticTrace) -> Diagnosis:
        start = time.perf_counter()
        choice = self.rng.choice(self.candidate_causes)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return Diagnosis(
            trial_id=trace.trial_id,
            diagnostic_name=self.name,
            predicted_cause=choice,
            confidence=1.0 / len(self.candidate_causes),
            abstained=False,
            evidence_rationale=f"Random choice among {self.candidate_causes}",
            runtime_ms=round(elapsed_ms, 3),
        )
