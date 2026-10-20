"""Pytest configuration and shared test fixtures."""

import json
from pathlib import Path
from typing import Any

import pytest

from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord, RetrievedPassage


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def synthetic_fixture_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "synthetic_qa.json"


@pytest.fixture
def synthetic_qa_data(synthetic_fixture_path: Path) -> list[dict[str, Any]]:
    with open(synthetic_fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_diagnostic_trace() -> DiagnosticTrace:
    passages = [
        RetrievedPassage(
            passage_id="doc_prague",
            rank=1,
            score=9.5,
            text="Prague is situated on the Vltava River in central Bohemia.",
        ),
        RetrievedPassage(
            passage_id="doc_kafka",
            rank=2,
            score=8.2,
            text="Franz Kafka was born into a Jewish family in Prague.",
        ),
    ]
    return DiagnosticTrace(
        trial_id="trial_sample_001",
        question_id="syn_q001",
        question="Which river flows through the birthplace of the author of 'The Metamorphosis'?",
        retrieved_top_k=passages,
        candidate_pool_scores=[9.5, 8.2, 5.0, 3.1, 1.2],
        pre_assembly_passage_ids=["doc_prague", "doc_kafka"],
        post_assembly_passage_ids=["doc_prague", "doc_kafka"],
        generator_context="Prague is on Vltava River. Kafka was born in Prague.",
        answer="Vltava River",
        latency_ms=15.4,
    )


@pytest.fixture
def sample_evaluator_record() -> EvaluatorRecord:
    return EvaluatorRecord(
        trial_id="trial_sample_001",
        question_id="syn_q001",
        split="dev",
        injected_fault="F0",
        fault_parameters={},
        gold_answers=["Vltava River"],
        gold_evidence_ids=["Franz Kafka", "Prague"],
        injection_applied=False,
        trace_condition_achieved=True,
        trial_valid=True,
        answer_exact_match=True,
        answer_token_f1=1.0,
        answer_supported_by_context=True,
    )
