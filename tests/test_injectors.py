"""Unit and contract tests for fault injectors F1–F4."""

from pathlib import Path
from rag_autopsy.core.schemas import RetrievedPassage
from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.faults.f1_unavailable import F1EvidenceUnavailableInjector
from rag_autopsy.faults.f2_ranked_out import F2EvidenceRankedOutInjector
from rag_autopsy.faults.f3_assembly_loss import F3ContextAssemblyLossInjector
from rag_autopsy.faults.f4_contradiction import F4ContradictoryEvidenceInjector
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.retriever import BM25Retriever


def test_f1_injector(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)
    corpus = Corpus.from_qa_records(records)
    rec = records[0]

    injector = F1EvidenceUnavailableInjector()
    res = injector.inject(rec, corpus)

    assert res.injected_fault == "F1"
    assert res.injection_applied is True
    assert res.trace_condition_achieved is True
    assert res.trial_valid is True

    # Base corpus should still contain all passages
    assert len(corpus) == 20
    # Trial corpus should have gold passages removed
    trial_corpus = res.modified_corpus  # type: ignore[attr-defined]
    assert len(trial_corpus) == 18
    for gid in rec.gold_evidence_passage_ids:
        assert trial_corpus.get_passage(gid) is None


def test_f2_injector(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)
    corpus = Corpus.from_qa_records(records)
    rec = records[0]

    injector = F2EvidenceRankedOutInjector(distractor_count=5, cutoff_k=3)
    res = injector.inject(rec, corpus)

    assert res.injected_fault == "F2"
    assert res.injection_applied is True
    assert res.trace_condition_achieved is True
    assert res.trial_valid is True

    trial_corpus = res.modified_corpus  # type: ignore[attr-defined]
    # Gold passages must still exist in the trial corpus
    for gid in rec.gold_evidence_passage_ids:
        assert trial_corpus.get_passage(gid) is not None

    # Gold passages must NOT be in top-3
    retriever = BM25Retriever(trial_corpus)
    top_k, _ = retriever.retrieve(rec.question, top_k=3)
    top_ids = [p.passage_id for p in top_k]
    for gid in rec.gold_evidence_passage_ids:
        assert gid not in top_ids


def test_f3_injector(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)
    rec = records[0]

    gold_id = rec.gold_evidence_passage_ids[0]
    passages = [
        RetrievedPassage(passage_id=gold_id, rank=1, score=10.0, text="Gold passage"),
        RetrievedPassage(passage_id="distractor_1", rank=2, score=5.0, text="Distractor"),
    ]

    injector = F3ContextAssemblyLossInjector()
    surviving, res = injector.apply_to_retrieved_passages(passages, [gold_id])

    assert res.injected_fault == "F3"
    assert res.injection_applied is True
    assert res.trace_condition_achieved is True
    assert res.trial_valid is True

    surviving_ids = [p.passage_id for p in surviving]
    assert gold_id not in surviving_ids
    assert "distractor_1" in surviving_ids


def test_f3_ineffective_when_gold_not_retrieved() -> None:
    passages = [
        RetrievedPassage(passage_id="distractor_1", rank=1, score=5.0, text="Distractor"),
    ]
    injector = F3ContextAssemblyLossInjector()
    _, res = injector.apply_to_retrieved_passages(passages, ["nonexistent_gold"])

    assert res.injection_applied is False
    assert res.trial_valid is False
    assert "Precondition failed" in str(res.invalid_reason)


def test_f4_injector(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)
    rec = records[0]

    gold_id = rec.gold_evidence_passage_ids[0]
    passages = [
        RetrievedPassage(passage_id=gold_id, rank=1, score=10.0, text="Gold passage content"),
    ]

    injector = F4ContradictoryEvidenceInjector()
    modified, res = injector.apply_to_context_passages(rec, passages)

    assert res.injected_fault == "F4"
    assert res.injection_applied is True
    assert res.trace_condition_achieved is True
    assert res.trial_valid is True

    # Both gold and contradiction exist in modified list
    modified_ids = [p.passage_id for p in modified]
    assert gold_id in modified_ids
    assert any("f4_contradiction" in pid for pid in modified_ids)
