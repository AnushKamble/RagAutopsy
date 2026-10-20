"""Pluggable generator interfaces with hash-keyed disk caching and real model adapters."""

import hashlib
import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseGenerator(ABC):
    """Abstract base class for all RAG generation adapters."""

    is_test_generator: bool = False

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Produce an answer string given a question and assembled context."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the unique model or adapter name."""

    def get_config(self) -> dict[str, Any]:
        """Return all result-changing configuration parameters for caching."""
        return {}


class MockExtractiveGenerator(BaseGenerator):
    """Deterministic, heuristic generator used STRICTLY FOR TESTING INFRASTRUCTURE.
    
    WARNING: This is a test harness fixture and must NEVER be reported as a real generative
    baseline. It performs simple sentence-level word overlap extraction.
    """

    is_test_generator: bool = True

    def __init__(self, fallback_abstention: str = "Insufficient context to answer the question."):
        self.fallback_abstention = fallback_abstention

    def get_model_name(self) -> str:
        return "test_extractive_heuristic"

    def get_config(self) -> dict[str, Any]:
        return {
            "strategy": "sentence_word_overlap",
            "test_infrastructure_only": True,
            "fallback_abstention": self.fallback_abstention,
        }

    def generate(self, question: str, context: str) -> str:
        if not context or not context.strip():
            return self.fallback_abstention

        # Split context into clean sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        if not sentences:
            return self.fallback_abstention

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
    """Caching wrapper around any generator, strictly keyed by all result-changing inputs."""

    def __init__(self, inner_generator: BaseGenerator, cache_dir: str | None = ".cache/generator"):
        self.inner_generator = inner_generator
        self.is_test_generator = getattr(inner_generator, "is_test_generator", False)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_name(self) -> str:
        return f"cached_{self.inner_generator.get_model_name()}"

    def get_config(self) -> dict[str, Any]:
        return self.inner_generator.get_config()

    def compute_cache_key(self, question: str, context: str) -> str:
        """Compute SHA-256 hash strictly keyed by model name, config params, question, and context."""
        model_name = self.inner_generator.get_model_name()
        config_canonical = json.dumps(self.inner_generator.get_config(), sort_keys=True)
        payload = f"{model_name}:::{config_canonical}:::{question.strip()}:::{context.strip()}".encode()
        return hashlib.sha256(payload).hexdigest()

    def generate(self, question: str, context: str) -> str:
        if not self.cache_dir:
            return self.inner_generator.generate(question, context)

        cache_key = self.compute_cache_key(question, context)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.is_file():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data["answer"]
            except (json.JSONDecodeError, KeyError):
                pass

        # Cache miss: execute generation and write cache
        answer = self.inner_generator.generate(question, context)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generator": self.inner_generator.get_model_name(),
                    "config": self.inner_generator.get_config(),
                    "cache_key": cache_key,
                    "question": question,
                    "context_hash": hashlib.sha256(context.encode("utf-8")).hexdigest(),
                    "answer": answer,
                },
                f,
                indent=2,
            )

        return answer


class HuggingFaceGenerator(BaseGenerator):
    """Local CPU-first HuggingFace transformers generator (e.g. Flan-T5, Qwen)."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self._pipeline = None

    def get_model_name(self) -> str:
        return f"hf_{self.model_name.replace('/', '_')}"

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "device": self.device,
        }

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "HuggingFaceGenerator requires 'transformers' and 'torch'. "
                "Install them with `pip install rag-autopsy[nli]` or `pip install transformers torch`."
            )

        task = "text2text-generation" if "t5" in self.model_name.lower() else "text-generation"
        self._pipeline = pipeline(
            task=task,
            model=self.model_name,
            device=-1 if self.device == "cpu" else 0,
        )
        return self._pipeline

    def generate(self, question: str, context: str) -> str:
        pipe = self._load_pipeline()
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on the context above. If the context does not contain the answer, say 'unsupported':\nAnswer:"
        )

        gen_kwargs = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        outputs = pipe(prompt, **gen_kwargs)
        if isinstance(outputs, list) and len(outputs) > 0:
            raw_text = outputs[0].get("generated_text", "")
            # If text-generation, strip the prompt prefix
            if prompt in raw_text:
                raw_text = raw_text.split("Answer:")[-1]
            return raw_text.strip()
        return "unsupported"


class OpenAICompatibleGenerator(BaseGenerator):
    """Generator adapter for local or remote OpenAI-compatible HTTP endpoints (e.g., Ollama, vLLM, LM-Studio).
    
    Uses standard library urllib.request to avoid external dependency requirements.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:11434/v1",
        model_name: str = "qwen2.5:0.5b",
        temperature: float = 0.0,
        max_tokens: int = 64,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or "EMPTY"
        self.timeout_seconds = timeout_seconds

    def get_model_name(self) -> str:
        return f"api_{self.model_name.replace(':', '_')}"

    def get_config(self) -> dict[str, Any]:
        return {
            "api_base": self.api_base,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def generate(self, question: str, context: str) -> str:
        endpoint = f"{self.api_base}/chat/completions"
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer the question directly and concisely based strictly on the provided context."
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a concise question-answering assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                return resp_json["choices"][0]["message"]["content"].strip()
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Failed to connect to OpenAI-compatible endpoint at {endpoint}: {e}. "
                "Ensure local model server (e.g. Ollama) is running."
            )
