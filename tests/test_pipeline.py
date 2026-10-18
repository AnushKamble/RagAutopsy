"""Integration and unit tests for the RAG reference pipeline."""

from pathlib import Path
from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import CachedGenerator, MockExtractiveGenerator
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever


def test_corpus_creation_and_trial_views(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)

    corpus = Corpus.from_qa_records(records)
    assert len(corpus) == 20  # 10 paragraphs * 2 questions

    first_p = corpus.passages[0]
    assert corpus.get_passage(first_p.passage_id) is not None

    # Trial view excluding one passage
    trial_view = corpus.without_passages({first_p.passage_id})
    assert len(trial_view) == 19
    assert trial_view.get_passage(first_p.passage_id) is None
    # Base corpus remains untouched
    assert len(corpus) == 20


def test_bm25_retrieval(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)
    corpus = Corpus.from_qa_records(records)

    retriever = BM25Retriever(corpus)
    query = "Which river flows through Prague and Kafka's birthplace?"

    top_passages, candidate_scores = retriever.retrieve(query, top_k=3, candidate_pool_size=10)
    assert len(top_passages) == 3
    assert len(candidate_scores) == 10

    # Scores should be in non-increasing order
    assert candidate_scores == sorted(candidate_scores, reverse=True)

    # Top passage should be relevant to Prague or Kafka
    top_titles = [p.text.split("\n")[0] for p in top_passages]
    assert any(t in ["Prague", "Franz Kafka", "Vltava"] for t in top_titles)


def test_context_builder_budget_truncation() -> None:
    builder = ContextBuilder(max_context_chars=120)  # very small budget

    from rag_autopsy.core.schemas import RetrievedPassage
    p1 = RetrievedPassage(passage_id="p1", rank=1, score=10.0, text="First passage text content.")
    p2 = RetrievedPassage(passage_id="p2", rank=2, score=8.0, text="Second passage text content.")
    p3 = RetrievedPassage(passage_id="p3", rank=3, score=6.0, text="Third passage text content.")

    context, pre_ids, post_ids = builder.build_context([p1, p2, p3])
    assert pre_ids == ["p1", "p2", "p3"]
    # Due to small budget, not all passages survive
    assert len(post_ids) < 3
    assert "p1" in post_ids


def test_cached_generator(tmp_path: Path) -> None:
    inner = MockExtractiveGenerator()
    cached = CachedGenerator(inner_generator=inner, cache_dir=str(tmp_path / "cache"))

    q = "What is the capital of France?"
    ctx = "Paris is the capital of France."

    ans1 = cached.generate(q, ctx)
    assert "Paris is the capital of France." in ans1

    # Verify cache file was created
    cache_files = list((tmp_path / "cache").glob("*.json"))
    assert len(cache_files) == 1

    # Second call hits cache
    ans2 = cached.generate(q, ctx)
    assert ans2 == ans1


def test_end_to_end_rag_pipeline(synthetic_fixture_path: Path, tmp_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)
    corpus = Corpus.from_qa_records(records)

    retriever = BM25Retriever(corpus)
    builder = ContextBuilder(max_context_chars=2000)
    generator = CachedGenerator(MockExtractiveGenerator(), cache_dir=str(tmp_path / "gen_cache"))

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=builder,
        generator=generator,
        top_k=3,
        candidate_pool_size=10,
    )

    rec = records[0]
    trace, eval_rec = pipeline.run(
        trial_id="clean_trial_001",
        question_id=rec.question_id,
        question=rec.question,
        split="dev",
        gold_answers=[rec.gold_answer],
        gold_evidence_ids=rec.gold_evidence_passage_ids,
        injected_fault="F0",
    )

    # Verify DiagnosticTrace structure
    assert trace.trial_id == "clean_trial_001"
    assert trace.question_id == rec.question_id
    assert len(trace.retrieved_top_k) == 3
    assert len(trace.candidate_pool_scores) == 10
    assert trace.answer != ""
    assert trace.latency_ms > 0

    # Verify EvaluatorRecord structure
    assert eval_rec.injected_fault == "F0"
    assert eval_rec.trial_valid is True
    assert eval_rec.answer_exact_match is not None
    assert eval_rec.answer_token_f1 is not None
