"""Pluggable generator interfaces with hash-keyed disk caching."""

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
import re
from typing import Optional


class BaseGenerator(ABC):
    """Abstract base class for all RAG generation adapters."""

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Produce an answer string given a question and assembled context."""
        pass


class MockExtractiveGenerator(BaseGenerator):
    """Deterministic, heuristic generator used for testing infrastructure on CPU.
    
    Extracts high-overlap sentence spans from context or returns an explicit
    abstention string if context is empty. Clearly marked as not a full LLM.
    """

    def generate(self, question: str, context: str) -> str:
        if not context or not context.strip():
            return "Insufficient context to answer the question."

        # Split context into clean sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        if not sentences:
            return "Insufficient context to answer the question."

        q_words = set(re.sub(r"[^\w\s]", "", question.lower()).split())

        best_sentence = ""
        best_score = -1.0

        for sent in sentences:
            # Skip passage headers like "[Passage 1]:"
            if sent.startswith("[Passage"):
                continue
            s_words = set(re.sub(r"[^\w\s]", "", sent.lower()).split())
            overlap = len(q_words.intersection(s_words))
            if overlap > best_score:
                best_score = overlap
                best_sentence = sent

        if best_sentence and best_score > 0:
            return best_sentence
        return sentences[0]


class CachedGenerator(BaseGenerator):
    """Caching wrapper around any generator, keyed by prompt, context, and generator config."""

    def __init__(self, inner_generator: BaseGenerator, cache_dir: Optional[str] = ".cache/generator"):
        self.inner_generator = inner_generator
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, question: str, context: str) -> str:
        gen_name = self.inner_generator.__class__.__name__
        payload = f"{gen_name}:::{question.strip()}:::{context.strip()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def generate(self, question: str, context: str) -> str:
        if not self.cache_dir:
            return self.inner_generator.generate(question, context)

        cache_key = self._compute_key(question, context)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.is_file():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data["answer"]
            except (json.JSONDecodeError, KeyError):
                pass

        # Cache miss: generate and save
        answer = self.inner_generator.generate(question, context)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generator": self.inner_generator.__class__.__name__,
                    "cache_key": cache_key,
                    "question": question,
                    "answer": answer,
                },
                f,
                indent=2,
            )

        return answer
