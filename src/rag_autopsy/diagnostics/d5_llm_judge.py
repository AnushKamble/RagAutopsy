"""LLM-as-a-judge diagnostic baseline evaluating unprivileged trace attribution."""

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from rag_autopsy.core.schemas import Diagnosis, DiagnosticTrace
from rag_autopsy.diagnostics.base import BaseDiagnostic


class D5LLMJudgeDiagnostic(BaseDiagnostic):
    """Diagnostic that prompts a local or remote LLM (e.g. Qwen 3 8B via Ollama) to attribute RAG failures.

    Operates strictly on unprivileged `DiagnosticTrace` fields, presenting the same observable
    evidence as D1-D4 (query, top-k passages, ranks, BM25 scores, pre/post assembly IDs, context, answer).
    Caches diagnoses by trace content hash to prevent redundant LLM inferences.
    """

    VALID_CAUSES = {"F0", "F1", "F2", "F3", "F4", "F5", "UNKNOWN"}

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        api_base: str = "http://localhost:11434/v1",
        cache_dir: str | None = ".cache/llm_judge",
        timeout_seconds: float = 45.0,
        temperature: float = 0.0,
    ):
        super().__init__(name="d5_llm_judge")
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_prompt(self, trace: DiagnosticTrace) -> str:
        """Construct structured unprivileged diagnosis prompt."""
        passages_summary = [
            f"- Rank {p.rank} (Score: {p.score:.2f}, ID: {p.passage_id}): {p.text[:140]}..."
            for p in trace.retrieved_top_k[:5]
        ]
        passages_text = "\n".join(passages_summary) if passages_summary else "No passages retrieved."

        return f"""You are an expert diagnostic system analyzing an unprivileged execution trace of a RAG pipeline.

Query: "{trace.question}"

Retrieved Passages:
{passages_text}

Pre-Assembly Passage IDs: {trace.pre_assembly_passage_ids}
Post-Assembly Passage IDs: {trace.post_assembly_passage_ids}

Context Assembled:
\"\"\"{trace.generator_context[:400]}...\"\"\"

Generated Answer:
\"{trace.answer}\"

Failure Taxonomy:
- F0: Clean execution (retrieval succeeded, context preserved, answer grounded).
- F1: Evidence Unavailable (relevant information completely missing from corpus).
- F2: Ranked Out (relevant passage retrieved below top-k cutoff).
- F3: Context Assembly Loss (passage present in pre-assembly was dropped during post-assembly).
- F4: Contradictory Evidence (conflicting distractor in context corrupted answer).
- F5: Unsupported Generation / Hallucination (context has valid evidence, but generator hallucinated).
- UNKNOWN: Ambiguous or insufficient observable trace signals.

Classify the primary failure stage into exactly ONE of the taxonomy codes.
Return strictly a JSON object:
{{"predicted_cause": "<F0|F1|F2|F3|F4|F5|UNKNOWN>", "confidence": <float 0.0-1.0>, "rationale": "<concise explanation>"}}"""

    def _compute_cache_key(self, trace: DiagnosticTrace, prompt: str) -> str:
        payload = f"{self.model_name}:::{prompt}:::{trace.answer}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _predict(self, trace: DiagnosticTrace) -> Diagnosis:
        import time

        start = time.perf_counter()
        prompt = self._build_prompt(trace)
        cache_key = self._compute_cache_key(trace, prompt)

        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.is_file():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        elapsed_ms = (time.perf_counter() - start) * 1000.0
                        return Diagnosis(
                            trial_id=trace.trial_id,
                            diagnostic_name=self.name,
                            predicted_cause=data["predicted_cause"],
                            confidence=data.get("confidence", 0.8),
                            abstained=data.get("predicted_cause") == "UNKNOWN",
                            abstention_reason=data.get("abstention_reason"),
                            evidence_rationale=data.get("evidence_rationale", "Loaded from LLM judge cache."),
                            runtime_ms=round(elapsed_ms, 3),
                        )
                except (json.JSONDecodeError, KeyError):
                    pass

        # Execute LLM call
        endpoint = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise automated diagnostic system that responds only in JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 300,
            "stream": False,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        }
        req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                choice = resp_json["choices"][0]["message"]
                raw_text = choice.get("content", "").strip()
                if not raw_text and "reasoning" in choice:
                    raw_text = choice["reasoning"].strip()

                parsed = self._parse_json_response(raw_text)
                pred = parsed.get("predicted_cause", "UNKNOWN").upper().strip()
                if pred not in self.VALID_CAUSES:
                    match = re.search(r"\b(F[0-5]|UNKNOWN)\b", pred)
                    pred = match.group(1) if match else "UNKNOWN"

                conf = float(parsed.get("confidence", 0.8))
                rationale = parsed.get("rationale", "Diagnosed by LLM judge.")
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                is_abstained = pred == "UNKNOWN"

                diagnosis = Diagnosis(
                    trial_id=trace.trial_id,
                    diagnostic_name=self.name,
                    predicted_cause=pred,
                    confidence=min(max(conf, 0.0), 1.0),
                    abstained=is_abstained,
                    abstention_reason="LLM judge abstained with UNKNOWN" if is_abstained else None,
                    evidence_rationale=f"[{self.model_name}] {rationale}",
                    runtime_ms=round(elapsed_ms, 3),
                )

                if self.cache_dir:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "predicted_cause": diagnosis.predicted_cause,
                                "confidence": diagnosis.confidence,
                                "evidence_rationale": diagnosis.evidence_rationale,
                                "raw_response": raw_text,
                            },
                            f,
                            indent=2,
                        )

                return diagnosis

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return Diagnosis(
                trial_id=trace.trial_id,
                diagnostic_name=self.name,
                predicted_cause="UNKNOWN",
                confidence=0.0,
                abstained=True,
                abstention_reason=f"LLM judge unreachable: {e}",
                evidence_rationale=f"LLM judge unreachable or failed ({e}); abstaining safely.",
                runtime_ms=round(elapsed_ms, 3),
            )

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Robustly extract JSON dictionary from text, handling markdown blocks."""
        # Check for ```json ... ``` blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw string
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try finding outermost braces
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except json.JSONDecodeError:
                pass

        # Heuristic fallback if JSON fails
        code_match = re.search(r"\b(F[0-5]|UNKNOWN)\b", text)
        if code_match:
            return {
                "predicted_cause": code_match.group(1),
                "confidence": 0.7,
                "rationale": "Extracted from non-JSON text.",
            }

        return {"predicted_cause": "UNKNOWN", "confidence": 0.0, "rationale": "Failed to parse JSON."}
