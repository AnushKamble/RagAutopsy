"""Unit tests for Combined / Compound Fault Injector (F2+F3, F3+F4)."""

from pathlib import Path

from rag_autopsy.core.schemas import generate_stable_trial_id
from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.faults.combined import CombinedFaultInjector
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import MockExtractiveGenerator
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever


def test_combined_f2_f3_injector(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)
    corpus = Corpus.from_qa_records(records)
    # rec[1] has gold evidence retrieved in top-k
    rec = records[1]

    comb_inj = CombinedFaultInjector(fault_types=["F2", "F3"], distractor_count=2, cutoff_k=5)
    assert comb_inj.combination_id == "F2+F3"

    # Step 1: inject corpus stage (F2)
    corpus_res = comb_inj.inject(rec, corpus)
    assert corpus_res.injected_fault == "F2+F3"
    assert corpus_res.injection_applied is True

    # Step 2: pipeline execution
    pipeline = RAGPipeline(
        retriever=BM25Retriever(corpus_res.modified_corpus or corpus),
        context_builder=ContextBuilder(max_context_chars=3000),
        generator=MockExtractiveGenerator(),
        top_k=5,
    )
    opaque_trial_id = generate_stable_trial_id(seed=42, question_id=rec.question_id, intervention_key="comb_f2_f3")

    trace, ev = comb_inj.execute_trial(
        trial_id=opaque_trial_id,
        qa_record=rec,
        base_corpus=corpus,
        pipeline=pipeline,
    )

    # Verification: trace has valid structure and schema
    assert trace.trial_id == opaque_trial_id
    assert trace.question_id == rec.question_id
    assert len(trace.retrieved_top_k) > 0

    # Evaluator record has compound fault details
    assert ev.injected_fault == "F2+F3"
    assert ev.injection_applied is True
    assert ev.trial_valid is True
    assert "corpus_stage" in ev.fault_parameters
    assert "passage_stage" in ev.fault_parameters


def test_combined_f3_f4_injector(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)
    corpus = Corpus.from_qa_records(records)
    # rec[1] has gold evidence retrieved in top-k
    rec = records[1]

    comb_inj = CombinedFaultInjector(fault_types=["F3", "F4"])
    assert comb_inj.combination_id == "F3+F4"

    pipeline = RAGPipeline(
        retriever=BM25Retriever(corpus),
        context_builder=ContextBuilder(max_context_chars=3000),
        generator=MockExtractiveGenerator(),
        top_k=5,
    )
    opaque_trial_id = generate_stable_trial_id(seed=42, question_id=rec.question_id, intervention_key="comb_f3_f4")

    trace, ev = comb_inj.execute_trial(
        trial_id=opaque_trial_id,
        qa_record=rec,
        base_corpus=corpus,
        pipeline=pipeline,
    )

    assert trace.trial_id == opaque_trial_id
    assert ev.injected_fault == "F3+F4"
    assert ev.injection_applied is True
    assert ev.trial_valid is True


def test_combined_injector_precondition_rejection_when_unretrieved(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)
    corpus = Corpus.from_qa_records(records)
    # rec[0] has no gold evidence retrieved in top-3 by clean BM25
    rec = records[0]

    comb_inj = CombinedFaultInjector(fault_types=["F3", "F4"])
    pipeline = RAGPipeline(
        retriever=BM25Retriever(corpus),
        context_builder=ContextBuilder(max_context_chars=3000),
        generator=MockExtractiveGenerator(),
        top_k=3,
    )
    opaque_trial_id = generate_stable_trial_id(seed=42, question_id=rec.question_id, intervention_key="comb_precond_rej")

    trace, ev = comb_inj.execute_trial(
        trial_id=opaque_trial_id,
        qa_record=rec,
        base_corpus=corpus,
        pipeline=pipeline,
    )

    # Demonstrates accurate adjudication: invalid trials are flagged cleanly
    assert ev.trial_valid is False
    assert "Precondition failed" in str(ev.invalid_reason)
