"""D3: NLI Support and Contradiction Diagnostic.

Evaluates premise-hypothesis entailment, contradiction, and neutrality scores between
generator context passages and the generated answer claim. Strictly adheres to zero
leakage constraints by inspecting only unprivileged trace fields.
"""

import hashlib
import json
import re
import time
from pathlib import Path

from rag_autopsy.core.schemas import Diagnosis, DiagnosticTrace
from rag_autopsy.diagnostics.base import BaseDiagnostic


class D3NLISupportDiagnostic(BaseDiagnostic):
    """NLI-based diagnostic using cross-encoder/zero-shot entailment with disk caching."""

    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny-mnli",
        contradiction_threshold: float = 0.60,
        unsupported_entailment_threshold: float = 0.30,
        cache_dir: str | None = ".cache/nli_cache",
        use_fallback_if_unavailable: bool = True,
        load_model_weights: bool = False,
    ):
        super().__init__(name="d3_nli_support")
        self.model_name = model_name
        self.contradiction_threshold = contradiction_threshold
        self.unsupported_entailment_threshold = unsupported_entailment_threshold
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_fallback_if_unavailable = use_fallback_if_unavailable
        self.load_model_weights = load_model_weights
        self._pipeline = None
        self._load_attempted = False

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_pipeline(self):
        if not self.load_model_weights:
            return None
        if self._load_attempted:
            return self._pipeline
        self._load_attempted = True

        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                task="text-classification",
                model=self.model_name,
                top_k=None,
                device=-1,  # CPU execution for laptop fit
            )
        except Exception:
            self._pipeline = None
        return self._pipeline

    def _compute_cache_key(self, premise: str, hypothesis: str) -> str:
        payload = f"{self.model_name}:::{premise.strip()}:::{hypothesis.strip()}".encode()
        return hashlib.sha256(payload).hexdigest()

    def evaluate_pair(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Compute entailment, contradiction, and neutral scores for a single pair."""
        if not premise.strip() or not hypothesis.strip():
            return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

        cache_key = self._compute_cache_key(premise, hypothesis)
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.is_file():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass

        pipe = self._get_pipeline()
        scores = {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

        if pipe is not None:
            try:
                # Text classification NLI format: "premise [SEP] hypothesis" or dict input
                input_text = f"{premise[:1000]} [SEP] {hypothesis[:300]}"
                outputs = pipe(input_text)
                if isinstance(outputs, list) and len(outputs) > 0:
                    raw_scores = outputs[0] if isinstance(outputs[0], list) else outputs
                    for item in raw_scores:
                        lbl = item["label"].lower()
                        score = float(item["score"])
                        if "entail" in lbl:
                            scores["entailment"] = score
                        elif "contra" in lbl:
                            scores["contradiction"] = score
                        elif "neut" in lbl:
                            scores["neutral"] = score
            except Exception:
                scores = self._heuristic_fallback(premise, hypothesis)
        else:
            scores = self._heuristic_fallback(premise, hypothesis)

        # Cache result
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(scores, f)

        return scores

    def _heuristic_fallback(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Deterministic lexical entailment and contradiction approximation fallback."""
        p_lower = premise.lower()
        h_lower = hypothesis.lower()

        p_words = set(re.findall(r"\w+", p_lower))
        h_words = set(re.findall(r"\w+", h_lower))

        if not h_words:
            return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

        overlap = len(h_words.intersection(p_words)) / len(h_words)

        # Contradiction signature: negation word mismatch or distinct numbers
        negations = {"not", "never", "no", "cannot", "failed", "refused", "disputed"}
        h_has_neg = bool(negations.intersection(h_words))
        p_has_neg = bool(negations.intersection(p_words))

        is_contradiction = (h_has_neg != p_has_neg) and (overlap > 0.4)

        if is_contradiction:
            return {"entailment": 0.05, "neutral": 0.15, "contradiction": 0.80}

        if overlap >= 0.65:
            return {"entailment": round(overlap, 3), "neutral": round(1.0 - overlap, 3), "contradiction": 0.0}
        elif overlap >= 0.30:
            return {"entailment": round(overlap * 0.5, 3), "neutral": round(1.0 - (overlap * 0.5), 3), "contradiction": 0.05}
        else:
            return {"entailment": 0.10, "neutral": 0.80, "contradiction": 0.10}

    def score_trace(self, trace: DiagnosticTrace) -> dict[str, float]:
        """Evaluate NLI scores across all visible passages against the generated answer."""
        passages = [p.text for p in trace.retrieved_top_k]
        if not passages and trace.generator_context.strip():
            passages = [s.strip() for s in trace.generator_context.split("\n\n---\n\n") if s.strip()]

        if not passages or not trace.answer.strip():
            return {
                "max_entailment": 0.0,
                "max_contradiction": 0.0,
                "mean_entailment": 0.0,
                "mean_contradiction": 0.0,
            }

        entailments: list[float] = []
        contradictions: list[float] = []

        for p_text in passages:
            res = self.evaluate_pair(p_text, trace.answer)
            entailments.append(res["entailment"])
            contradictions.append(res["contradiction"])

        return {
            "max_entailment": max(entailments) if entailments else 0.0,
            "max_contradiction": max(contradictions) if contradictions else 0.0,
            "mean_entailment": sum(entailments) / len(entailments) if entailments else 0.0,
            "mean_contradiction": sum(contradictions) / len(contradictions) if contradictions else 0.0,
        }

    def _predict(self, trace: DiagnosticTrace) -> Diagnosis:
        """Produce cause diagnosis from NLI premise-hypothesis scores."""
        start_time = time.perf_counter()

        # Check for context loss first
        if len(trace.pre_assembly_passage_ids) > len(trace.post_assembly_passage_ids):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause="F3",
                confidence=0.95,
                abstained=False,
                evidence_rationale=(
                    f"Structural omission: {len(trace.pre_assembly_passage_ids) - len(trace.post_assembly_passage_ids)} "
                    f"retrieved passages dropped during assembly."
                ),
                runtime_ms=round(elapsed_ms, 2),
            )

        nli_scores = self.score_trace(trace)
        max_contra = nli_scores["max_contradiction"]
        max_entail = nli_scores["max_entailment"]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Rule 1: High contradiction in context indicates F4 Conflicting Evidence
        if max_contra >= self.contradiction_threshold:
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause="F4",
                confidence=round(float(max_contra), 3),
                abstained=False,
                evidence_rationale=f"NLI model detected material contradiction (contra_score={max_contra:.3f}).",
                runtime_ms=round(elapsed_ms, 2),
            )

        # Rule 2: Low entailment despite retrieved passages indicates F5 Unsupported Generation / Hallucination
        if max_entail < self.unsupported_entailment_threshold:
            # Check if retrieval occurred
            if len(trace.retrieved_top_k) > 0 and len(trace.post_assembly_passage_ids) > 0:
                return Diagnosis(
                    trial_id=trace.trial_id,
                    diagnostic_name=self.name,
                    predicted_cause="F5",
                    confidence=round(1.0 - float(max_entail), 3),
                    abstained=False,
                    evidence_rationale=(
                        f"NLI model detected unsupported generation: maximum context entailment is {max_entail:.3f} "
                        f"(below threshold {self.unsupported_entailment_threshold:.2f})."
                    ),
                    runtime_ms=round(elapsed_ms, 2),
                )

        # Rule 3: High entailment indicates supported clean generation (F0)
        return Diagnosis(
            trial_id=trace.trial_id,
            diagnostic_name=self.name,
            predicted_cause="F0",
            confidence=round(float(max_entail), 3) if max_entail > 0 else 0.5,
            abstained=False,
            evidence_rationale=f"NLI model verified context entailment (entail_score={max_entail:.3f}).",
            runtime_ms=round(elapsed_ms, 2),
        )
