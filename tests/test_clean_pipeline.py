"""Unit and contract tests for the clean RAG reference pipeline and trace completeness."""

import json
from pathlib import Path
import pytest

from rag_autopsy.core.schemas import (
    DiagnosticTrace,
    RetrievedPassage,
    verify_trace_reconstructibility,
)
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import (
    BaseGenerator,
    CachedGenerator,
    HuggingFaceGenerator,
    MockExtractiveGenerator,
    OpenAICompatibleGenerator,
)
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever


class DummyConfigurableGenerator(BaseGenerator):
    """Configurable generator for verifying result-changing cache key invariants."""
    def __init__(self, model_name: str = "dummy_v1", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    def get_model_name(self) -> str:
        return self.model_name

    def get_config(self):
        return {"model_name": self.model_name, "temperature": self.temperature}

    def generate(self, question: str, context: str) -> str:
        return f"Answer({self.model_name}, t={self.temperature}): {question}"


def test_mock_extractive_generator_is_explicitly_labeled_as_test_only():
    gen = MockExtractiveGenerator()
    assert gen.is_test_generator is True
    assert "test_extractive_heuristic" in gen.get_model_name()
    cfg = gen.get_config()
    assert cfg.get("test_infrastructure_only") is True


def test_cache_key_sensitive_to_all_result_changing_inputs(tmp_path: Path):
    gen_v1_t0 = DummyConfigurableGenerator(model_name="model_a", temperature=0.0)
    gen_v1_t1 = DummyConfigurableGenerator(model_name="model_a", temperature=0.7)
    gen_v2_t0 = DummyConfigurableGenerator(model_name="model_b", temperature=0.0)

    c1 = CachedGenerator(gen_v1_t0, cache_dir=str(tmp_path))
    c2 = CachedGenerator(gen_v1_t1, cache_dir=str(tmp_path))
    c3 = CachedGenerator(gen_v2_t0, cache_dir=str(tmp_path))

    q1 = "What is X?"
    q2 = "What is Y?"
    ctx1 = "X is 42."
    ctx2 = "X is 100."

    k_base = c1.compute_cache_key(q1, ctx1)

    # 1. Changing question changes key
    k_q2 = c1.compute_cache_key(q2, ctx1)
    assert k_base != k_q2

    # 2. Changing context changes key
    k_ctx2 = c1.compute_cache_key(q1, ctx2)
    assert k_base != k_ctx2

    # 3. Changing model config (temperature) changes key
    k_temp = c2.compute_cache_key(q1, ctx1)
    assert k_base != k_temp

    # 4. Changing model name changes key
    k_model = c3.compute_cache_key(q1, ctx1)
    assert k_base != k_model


def test_trace_reconstructibility_validation():
    p1 = RetrievedPassage(passage_id="p1", rank=1, score=12.5, text="Albert Einstein was a physicist.", title="Einstein")
    p2 = RetrievedPassage(passage_id="p2", rank=2, score=9.0, text="He developed the theory of relativity.", title="Relativity")

    # Valid trace: generator context contains text of all surviving passages
    valid_trace = DiagnosticTrace(
        trial_id="opaque_trial_100",
        question_id="q_1",
        question="Who was Einstein?",
        retrieved_top_k=[p1, p2],
        source_passage_ids=["p1", "p2"],
        pre_assembly_passage_ids=["p1", "p2"],
        post_assembly_passage_ids=["p1", "p2"],
        generator_context="[Passage 1]:\nAlbert Einstein was a physicist.\n\n---\n\n[Passage 2]:\nHe developed the theory of relativity.",
        answer="A physicist who developed relativity.",
        latency_ms=15.2,
        generator_model="test_extractive_heuristic",
        generator_config={"test": True},
    )

    is_valid, err = verify_trace_reconstructibility(valid_trace)
    assert is_valid is True
    assert err is None

    # Invalid trace: passage p2 text is missing from generator context
    invalid_trace = valid_trace.model_copy(update={
        "generator_context": "[Passage 1]:\nAlbert Einstein was a physicist."
    })
    is_valid_inv, err_inv = verify_trace_reconstructibility(invalid_trace)
    assert is_valid_inv is False
    assert "missing from generator_context" in err_inv

    # Invalid trace: unretrieved passage ID in post_assembly_passage_ids
    invalid_id_trace = valid_trace.model_copy(update={
        "post_assembly_passage_ids": ["p1", "unretrieved_p99"]
    })
    is_valid_id, err_id = verify_trace_reconstructibility(invalid_id_trace)
    assert is_valid_id is False
    assert "was not present in retrieved_top_k" in err_id


def test_real_generator_adapters_interface():
    # OpenAICompatibleGenerator configuration contract
    api_gen = OpenAICompatibleGenerator(
        api_base="http://localhost:11434/v1",
        model_name="llama3.2:1b",
        temperature=0.2,
        max_tokens=128,
    )
    assert api_gen.get_model_name() == "api_llama3.2_1b"
    cfg = api_gen.get_config()
    assert cfg["model_name"] == "llama3.2:1b"
    assert cfg["temperature"] == 0.2

    # HuggingFaceGenerator configuration contract
    hf_gen = HuggingFaceGenerator(
        model_name="google/flan-t5-small",
        max_new_tokens=32,
    )
    assert hf_gen.get_model_name() == "hf_google_flan-t5-small"
    assert hf_gen.get_config()["model_name"] == "google/flan-t5-small"


def test_clean_pipeline_instrumentation_completeness(synthetic_fixture_path: Path, tmp_path: Path):
    from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records

    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)
    corpus = Corpus.from_qa_records(records)
    rec = records[0]

    pipeline = RAGPipeline(
        retriever=BM25Retriever(corpus),
        context_builder=ContextBuilder(max_context_chars=2000),
        generator=CachedGenerator(MockExtractiveGenerator(), cache_dir=str(tmp_path)),
        top_k=3,
        candidate_pool_size=10,
    )

    trace, eval_rec = pipeline.run(
        trial_id="trial_verify_001",
        question_id=rec.question_id,
        question=rec.question,
        gold_answers=[rec.gold_answer],
        gold_evidence_ids=rec.gold_evidence_passage_ids,
        injected_fault="F0",
    )


    # 1. Ranks and scores instrumented
    assert len(trace.retrieved_top_k) == 3
    for p in trace.retrieved_top_k:
        assert p.rank in (1, 2, 3)
        assert isinstance(p.score, float)
        assert p.text != ""
        assert p.title is not None

    # 2. Source IDs
    assert len(trace.source_passage_ids) == 3
    assert trace.source_passage_ids == [p.passage_id for p in trace.retrieved_top_k]

    # 3. Context passage IDs
    assert trace.pre_assembly_passage_ids == trace.source_passage_ids
    assert len(trace.post_assembly_passage_ids) > 0

    # 4. Exact generator context
    assert trace.generator_context != ""
    assert "[Passage 1]:" in trace.generator_context

    # 5. Answer and runtime
    assert trace.answer != ""
    assert trace.latency_ms > 0

    # 6. Model and config metadata
    assert trace.generator_model is not None
    assert "test_extractive_heuristic" in trace.generator_model
    assert trace.generator_config.get("test_infrastructure_only") is True
    assert trace.metadata["top_k"] == 3

    # 7. Reconstructibility audit
    is_valid, err = verify_trace_reconstructibility(trace)
    assert is_valid is True
    assert err is None
