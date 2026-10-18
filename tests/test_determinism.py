"""Determinism, boundary robustness, and caching tests for RAG-Autopsy.

Spec Section 17, 18:
Verifies that:
1. Identical inputs and seeds produce bit-for-bit identical trial IDs, traces, and diagnoses.
2. Edge cases (empty retrieval, missing passages, long contexts) are handled gracefully and deterministically.
3. Generator disk cache produces exact reproducible outputs across multiple runs.
"""

from pathlib import Path
from rag_autopsy.core.schemas import DiagnosticTrace, generate_stable_trial_id
from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic
from rag_autopsy.diagnostics.d2_support import D2LexicalSupportDiagnostic
from rag_autopsy.diagnostics.d4_policy import D4CombinedAttributionPolicy
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import CachedGenerator, MockExtractiveGenerator
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever


def test_pipeline_trace_bit_for_bit_determinism(synthetic_fixture_path: Path) -> None:
    records = parse_hotpot_records(load_raw_hotpot_json(synthetic_fixture_path))
    corpus = Corpus.from_qa_records(records)
    rec = records[0]

    def run_trial(seed: int):
        pipe = RAGPipeline(
            retriever=BM25Retriever(corpus),
            context_builder=ContextBuilder(max_context_chars=3000),
            generator=MockExtractiveGenerator(),
            top_k=3,
        )
        t_id = generate_stable_trial_id(seed=seed, question_id=rec.question_id, intervention_key="f0_det")
        tr, ev = pipe.run(
            trial_id=t_id,
            question_id=rec.question_id,
            question=rec.question,
            split="test",
            gold_answers=[rec.gold_answer],
            gold_evidence_ids=rec.gold_evidence_passage_ids,
            injected_fault="F0",
        )
        return tr, ev

    tr1, ev1 = run_trial(seed=42)
    tr2, ev2 = run_trial(seed=42)

    # Identical trial IDs
    assert tr1.trial_id == tr2.trial_id
    assert ev1.trial_id == ev2.trial_id

    # Identical retrieved passage IDs and scores
    assert [p.passage_id for p in tr1.retrieved_top_k] == [p.passage_id for p in tr2.retrieved_top_k]
    assert [p.score for p in tr1.retrieved_top_k] == [p.score for p in tr2.retrieved_top_k]

    # Identical candidate pool scores
    assert tr1.candidate_pool_scores == tr2.candidate_pool_scores

    # Identical assembled context and answer
    assert tr1.generator_context == tr2.generator_context
    assert tr1.answer == tr2.answer


def test_diagnostic_decision_determinism(synthetic_fixture_path: Path) -> None:
    records = parse_hotpot_records(load_raw_hotpot_json(synthetic_fixture_path))
    corpus = Corpus.from_qa_records(records)
    rec = records[1]

    pipe = RAGPipeline(
        retriever=BM25Retriever(corpus),
        context_builder=ContextBuilder(max_context_chars=3000),
        generator=MockExtractiveGenerator(),
        top_k=3,
    )
    t_id = generate_stable_trial_id(seed=42, question_id=rec.question_id, intervention_key="d_det")
    tr, _ = pipe.run(
        trial_id=t_id,
        question_id=rec.question_id,
        question=rec.question,
        split="test",
        gold_answers=[rec.gold_answer],
        gold_evidence_ids=rec.gold_evidence_passage_ids,
        injected_fault="F0",
    )

    d1 = D1RetrievalDiagnostic()
    d2 = D2LexicalSupportDiagnostic()
    d4 = D4CombinedAttributionPolicy()

    # Repeated diagnostic evaluations
    res1 = d4.diagnose(tr)
    res2 = d4.diagnose(tr)

    assert res1.predicted_cause == res2.predicted_cause
    assert res1.confidence == res2.confidence
    assert res1.evidence_rationale == res2.evidence_rationale


def test_empty_retrieval_and_empty_context_robustness() -> None:
    # Completely empty corpus
    empty_corpus = Corpus(passages=[])
    retriever = BM25Retriever(empty_corpus)
    builder = ContextBuilder(max_context_chars=1000)
    gen = MockExtractiveGenerator()

    pipe = RAGPipeline(
        retriever=retriever,
        context_builder=builder,
        generator=gen,
        top_k=3,
    )

    tr, ev = pipe.run(
        trial_id="trial_0000000000000000",
        question_id="empty_q",
        question="What is nothing?",
        split="test",
        gold_answers=["nothing"],
        injected_fault="F0",
    )

    assert len(tr.retrieved_top_k) == 0
    assert tr.generator_context == ""
    assert ev.trial_valid is True
    assert not ev.answer_supported_by_context


def test_long_context_truncation_determinism(synthetic_fixture_path: Path) -> None:
    records = parse_hotpot_records(load_raw_hotpot_json(synthetic_fixture_path))
    corpus = Corpus.from_qa_records(records)
    retriever = BM25Retriever(corpus)

    # Restrict context budget to 200 chars
    tight_builder = ContextBuilder(max_context_chars=200)
    retrieved, _ = retriever.retrieve(records[0].question, top_k=5)

    ctx1, pre1, post1 = tight_builder.build_context(retrieved)
    ctx2, pre2, post2 = tight_builder.build_context(retrieved)

    assert ctx1 == ctx2
    assert pre1 == pre2
    assert post1 == post2
    assert len(ctx1) <= 200
    assert len(post1) < len(pre1)  # Later passages truncated by budget


def test_generator_cache_hit_determinism(tmp_path: Path) -> None:
    base_gen = MockExtractiveGenerator()
    cached_gen = CachedGenerator(inner_generator=base_gen, cache_dir=str(tmp_path))

    q = "What is the capital of France?"
    ctx = "Paris is the capital of France."

    ans1 = cached_gen.generate(q, ctx)
    # Check cache directory contains the cache entry
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1

    # Second call must hit cache and return exact string
    ans2 = cached_gen.generate(q, ctx)
    assert ans1 == ans2
