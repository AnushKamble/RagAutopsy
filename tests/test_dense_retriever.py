"""Tests for DenseRetriever and HybridRetriever."""

from rag_autopsy.data.schema import PassageRecord
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.retriever import (
    DenseRetriever,
    HybridRetriever,
)


def test_dense_retriever_ranking():
    passages = [
        PassageRecord(passage_id="p1", text="The Apollo program landed humans on the Moon.", sentences=["The Apollo program landed humans on the Moon."], title="Apollo"),
        PassageRecord(passage_id="p2", text="The culinary history of French pastry and croissants.", sentences=["The culinary history of French pastry and croissants."], title="Pastry"),
        PassageRecord(passage_id="p3", text="NASA space exploration missions to Mars and the Moon.", sentences=["NASA space exploration missions to Mars and the Moon."], title="NASA"),
    ]
    corpus = Corpus(passages=passages)
    dense = DenseRetriever(corpus)

    top_p, pool = dense.retrieve("Who landed on the Moon during the Apollo missions?", top_k=2)
    assert len(top_p) == 2
    assert top_p[0].passage_id in ["p1", "p3"]
    assert top_p[0].score >= top_p[1].score
    assert len(pool) == 3


def test_hybrid_retriever_fusion():
    passages = [
        PassageRecord(passage_id="p1", text="The Apollo program landed humans on the Moon.", sentences=["The Apollo program landed humans on the Moon."], title="Apollo"),
        PassageRecord(passage_id="p2", text="The culinary history of French pastry and croissants.", sentences=["The culinary history of French pastry and croissants."], title="Pastry"),
        PassageRecord(passage_id="p3", text="NASA lunar exploration and astronaut landings.", sentences=["NASA lunar exploration and astronaut landings."], title="NASA"),
    ]
    corpus = Corpus(passages=passages)
    hybrid = HybridRetriever(corpus, alpha=0.5)

    top_p, _pool = hybrid.retrieve("Apollo astronauts Moon landing", top_k=2)
    assert len(top_p) == 2
    assert top_p[0].passage_id in ["p1", "p3"]
    assert top_p[0].score >= top_p[1].score


def test_empty_corpus_dense_and_hybrid():
    empty_corpus = Corpus(passages=[])
    dense = DenseRetriever(empty_corpus)
    top_d, pool_d = dense.retrieve("Test query")
    assert top_d == []
    assert pool_d == []

    hybrid = HybridRetriever(empty_corpus)
    top_h, pool_h = hybrid.retrieve("Test query")
    assert top_h == []
    assert pool_h == []
