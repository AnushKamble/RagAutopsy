"""Tests for D5LLMJudgeDiagnostic."""

import pytest

from rag_autopsy.core.schemas import DiagnosticTrace, RetrievedPassage
from rag_autopsy.diagnostics.d5_llm_judge import D5LLMJudgeDiagnostic


def test_d5_llm_judge_initialization(tmp_path):
    judge = D5LLMJudgeDiagnostic(model_name="qwen3:8b", cache_dir=str(tmp_path))
    assert judge.name == "d5_llm_judge"
    assert judge.model_name == "qwen3:8b"


def test_d5_json_parsing():
    judge = D5LLMJudgeDiagnostic(cache_dir=None)

    # 1. Pure JSON
    p1 = judge._parse_json_response('{"predicted_cause": "F3", "confidence": 0.9, "rationale": "context loss"}')
    assert p1["predicted_cause"] == "F3"
    assert p1["confidence"] == 0.9

    # 2. Markdown fenced JSON
    fenced = """```json
    {
      "predicted_cause": "F4",
      "confidence": 0.85,
      "rationale": "contradiction found"
    }
    ```"""
    p2 = judge._parse_json_response(fenced)
    assert p2["predicted_cause"] == "F4"
    assert p2["confidence"] == 0.85

    # 3. Fallback on malformed JSON containing code
    text = "The issue appears to be F2 because the score dropped."
    p3 = judge._parse_json_response(text)
    assert p3["predicted_cause"] == "F2"


def test_d5_boundary_rejection():
    judge = D5LLMJudgeDiagnostic(cache_dir=None)
    with pytest.raises(TypeError, match="Data boundary violation"):
        judge.diagnose({"invalid": "dict"})


def test_d5_offline_fallback():
    # Invalid port to test graceful abstention
    judge = D5LLMJudgeDiagnostic(api_base="http://127.0.0.1:9999/v1", cache_dir=None, timeout_seconds=1.0)
    trace = DiagnosticTrace(
        trial_id="trial_test_001",
        question_id="q1",
        question="What is X?",
        retrieved_top_k=[
            RetrievedPassage(passage_id="p1", rank=1, score=10.0, text="Context text."),
        ],
        pre_assembly_passage_ids=["p1"],
        post_assembly_passage_ids=["p1"],
        generator_context="Context text.",
        answer="Answer.",
        latency_ms=10.0,
    )
    diag = judge.diagnose(trace)
    assert diag.predicted_cause == "UNKNOWN"
    assert "abstaining safely" in diag.evidence_rationale


def test_d5_live_ollama_call_if_available(tmp_path):
    from rag_autopsy.pipeline.generator import OllamaGenerator

    if not OllamaGenerator.is_available():
        pytest.skip("Ollama is not running locally with qwen3:8b")

    judge = D5LLMJudgeDiagnostic(model_name="qwen3:8b", cache_dir=str(tmp_path), timeout_seconds=30.0)
    trace = DiagnosticTrace(
        trial_id="trial_live_001",
        question_id="q_bubblegum",
        question="Where is Bubblegum Alley located in California?",
        retrieved_top_k=[
            RetrievedPassage(
                passage_id="p1",
                rank=1,
                score=12.5,
                text="Bubblegum Alley is located in downtown San Luis Obispo, California.",
            ),
            RetrievedPassage(
                passage_id="p2",
                rank=2,
                score=4.2,
                text="Distractor passage about another gum wall.",
            ),
        ],
        pre_assembly_passage_ids=["p1", "p2"],
        post_assembly_passage_ids=["p2"],  # p1 was dropped during assembly -> F3!
        generator_context="[Passage 2]: Distractor passage about another gum wall.",
        answer="I do not have enough information to locate Bubblegum Alley.",
        latency_ms=15.0,
    )
    diag = judge.diagnose(trace)
    assert diag.predicted_cause in ["F3", "F1", "F2", "UNKNOWN"]
    assert diag.runtime_ms > 0
    assert (tmp_path / f"{judge._compute_cache_key(trace, judge._build_prompt(trace))}.json").is_file()
