"""D2: Lexical and structural support diagnostic rules."""

import re
import string
import time
from typing import List, Set
from rag_autopsy.core.schemas import Diagnosis, DiagnosticTrace
from rag_autopsy.diagnostics.base import BaseDiagnostic


CONTRADICTION_CUES = [
    "contradict",
    "conflict",
    "contrary to",
    "dispute",
    "incorrect claim",
    "refute",
    "opposed to",
    "false claim",
    "denies",
]


def extract_keywords(text: str) -> Set[str]:
    """Extract lowercase non-punctuation keywords."""
    clean = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text.lower())
    stop_words = {"the", "a", "an", "is", "was", "and", "or", "of", "to", "in", "it", "that", "this"}
    return {w for w in clean.split() if len(w) > 2 and w not in stop_words}


class D2LexicalSupportDiagnostic(BaseDiagnostic):
    """Diagnoses failures using lexical context support and contradiction heuristics."""

    def __init__(
        self,
        name: str = "d2_lexical_support",
        min_overlap_ratio: float = 0.3,
    ):
        super().__init__(name=name)
        self.min_overlap_ratio = min_overlap_ratio

    def _predict(self, trace: DiagnosticTrace) -> Diagnosis:
        start = time.perf_counter()

        context_lower = trace.generator_context.lower()
        answer_lower = trace.answer.lower()

        # Signal 1: Context assembly loss (structural)
        pre_ids = set(trace.pre_assembly_passage_ids)
        post_ids = set(trace.post_assembly_passage_ids)
        if len(pre_ids) > len(post_ids):
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause="F3",
                confidence=0.90,
                abstained=False,
                evidence_rationale="Structural context assembly loss detected before generator.",
                runtime_ms=round(elapsed_ms, 3),
            )

        # Signal 2: Internal context contradiction
        # Check if context contains explicit contradiction cues
        contradiction_found = any(cue in context_lower for cue in CONTRADICTION_CUES)
        if contradiction_found:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause="F4",
                confidence=0.85,
                abstained=False,
                evidence_rationale="Material contradiction cues detected in generator context.",
                runtime_ms=round(elapsed_ms, 3),
            )

        # Signal 3: Lexical answer grounding in context
        ans_keywords = extract_keywords(answer_lower)
        ctx_keywords = extract_keywords(context_lower)

        if not ans_keywords:
            overlap_ratio = 1.0
        else:
            overlap = ans_keywords.intersection(ctx_keywords)
            overlap_ratio = len(overlap) / len(ans_keywords)

        if overlap_ratio < self.min_overlap_ratio:
            # Answer is unsupported by context
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause="F1",  # Support diagnostic attributes missing evidence
                confidence=0.70,
                abstained=False,
                evidence_rationale=(
                    f"Generated answer has low lexical overlap with context "
                    f"({round(overlap_ratio, 2)} < {self.min_overlap_ratio})."
                ),
                runtime_ms=round(elapsed_ms, 3),
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return Diagnosis(
            trial_id=trace.trial_id,
            diagnostic_name=self.name,
            predicted_cause="F0",
            confidence=0.75,
            abstained=False,
            evidence_rationale=f"Answer is well-grounded in generator context (overlap={round(overlap_ratio, 2)}).",
            runtime_ms=round(elapsed_ms, 3),
        )
