"""Unit tests for F5: Unsupported Generation / Hallucination injector."""

from pathlib import Path
from rag_autopsy.core.schemas import RetrievedPassage
from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.faults.f5_unsupported import F5UnsupportedGenerationInjector
from rag_autopsy.pipeline.corpus import Corpus


def test_f5_injector_effective_when_evidence_present(synthetic_fixture_path: Path):
    raw = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw)
    rec = records[0]
    corpus = Corpus.from_qa_records([rec])

    injector = F5UnsupportedGenerationInjector()
    retrieved = [
        RetrievedPassage(passage_id=rec.gold_evidence_passage_ids[0], rank=1, score=15.0, text="Gold evidence text."),
        RetrievedPassage(passage_id="distractor_01", rank=2, score=10.0, text="Distractor text."),
    ]
    context = "[Passage 1]: Gold evidence text.\n\n---\n\n[Passage 2]: Distractor text."

    ans, res = injector.apply_to_generation(rec, retrieved, context)
    assert res.injection_applied is True
    assert res.trace_condition_achieved is True
    assert res.trial_valid is True
    assert res.injected_fault == "F5"
    assert ans != ""
    assert rec.gold_answer not in ans or "UNKNOWN_FICTITIOUS_VARIANT" in ans or "disputed" in ans


def test_f5_injector_ineffective_when_evidence_absent(synthetic_fixture_path: Path):
    raw = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw)
    rec = records[0]

    injector = F5UnsupportedGenerationInjector()
    # Passages contain NO gold evidence
    retrieved = [
        RetrievedPassage(passage_id="distractor_01", rank=1, score=10.0, text="Distractor 1 text."),
        RetrievedPassage(passage_id="distractor_02", rank=2, score=8.0, text="Distractor 2 text."),
    ]
    context = "[Passage 1]: Distractor 1 text."

    ans, res = injector.apply_to_generation(rec, retrieved, context)
    assert res.injection_applied is False
    assert res.trial_valid is False
    assert "Precondition failed" in res.invalid_reason
