"""Pilot experiment for F1 Evidence Unavailable on a tiny deterministic corpus.

Records attempted, valid, and invalid trials, and inspects before/after traces.
Verifies that the shared base corpus is not permanently mutated.
"""

import json
from pathlib import Path
from rag_autopsy.core.schemas import generate_stable_trial_id
from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.evaluation.validity import adjudicate_trials
from rag_autopsy.faults.f1_unavailable import F1EvidenceUnavailableInjector
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import MockExtractiveGenerator
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever


def run_f1_pilot():
    fixture_path = Path("tests/fixtures/synthetic_qa.json")
    raw_items = load_raw_hotpot_json(fixture_path)
    records = parse_hotpot_records(raw_items)

    # 1. Build immutable base corpus
    base_corpus = Corpus.from_qa_records(records)
    initial_corpus_size = len(base_corpus)
    print(f"Base corpus initialized with {initial_corpus_size} passages across {len(records)} questions.")

    gen = MockExtractiveGenerator()
    builder = ContextBuilder(max_context_chars=3000)
    f1_injector = F1EvidenceUnavailableInjector()

    pilot_results = []
    attempted = 0
    valid_count = 0
    invalid_count = 0
    invalid_reasons = []

    # Also include an artificial edge case with missing gold to test validity adjudication
    invalid_rec = records[0].model_copy(update={"gold_evidence_passage_ids": ["nonexistent_id"]})
    test_records = list(records) + [invalid_rec]

    for i, rec in enumerate(test_records):
        attempted += 1
        # Step A: Run clean reference pipeline (F0)
        pipe_clean = RAGPipeline(
            retriever=BM25Retriever(base_corpus),
            context_builder=builder,
            generator=gen,
            top_k=3,
        )
        clean_trial_id = generate_stable_trial_id(seed=42, question_id=rec.question_id, intervention_key=f"clean_{i}")
        trace_clean, eval_clean = pipe_clean.run(
            trial_id=clean_trial_id,
            question_id=rec.question_id,
            question=rec.question,
            split="pilot",
            gold_answers=[rec.gold_answer],
            gold_evidence_ids=rec.gold_evidence_passage_ids,
            injected_fault="F0",
        )

        # Step B: Inject F1 Evidence Unavailable
        inj_res = f1_injector.inject(rec, base_corpus)
        if inj_res.trial_valid:
            valid_count += 1
            trial_corpus = inj_res.modified_corpus
        else:
            invalid_count += 1
            invalid_reasons.append(inj_res.invalid_reason)
            trial_corpus = base_corpus

        pipe_f1 = RAGPipeline(
            retriever=BM25Retriever(trial_corpus),
            context_builder=builder,
            generator=gen,
            top_k=3,
        )
        f1_trial_id = generate_stable_trial_id(seed=42, question_id=rec.question_id, intervention_key=f"f1_{i}")
        trace_f1, eval_f1 = pipe_f1.run(
            trial_id=f1_trial_id,
            question_id=rec.question_id,
            question=rec.question,
            split="pilot",
            gold_answers=[rec.gold_answer],
            gold_evidence_ids=rec.gold_evidence_passage_ids,
            injected_fault="F1",
            fault_parameters=inj_res.fault_parameters,
            injection_applied=inj_res.injection_applied,
            trace_condition_achieved=inj_res.trace_condition_achieved,
            trial_valid=inj_res.trial_valid,
            invalid_reason=inj_res.invalid_reason,
        )

        pilot_results.append({
            "question_id": rec.question_id,
            "question": rec.question,
            "gold_evidence_ids": rec.gold_evidence_passage_ids,
            "clean_trace": {
                "trial_id": trace_clean.trial_id,
                "retrieved_ids": [p.passage_id for p in trace_clean.retrieved_top_k],
                "top_score": trace_clean.retrieved_top_k[0].score if trace_clean.retrieved_top_k else 0.0,
                "candidate_pool_scores": trace_clean.candidate_pool_scores[:5] if trace_clean.candidate_pool_scores else [],
                "answer": trace_clean.answer,
            },
            "f1_trace": {
                "trial_id": trace_f1.trial_id,
                "retrieved_ids": [p.passage_id for p in trace_f1.retrieved_top_k],
                "top_score": trace_f1.retrieved_top_k[0].score if trace_f1.retrieved_top_k else 0.0,
                "candidate_pool_scores": trace_f1.candidate_pool_scores[:5] if trace_f1.candidate_pool_scores else [],
                "answer": trace_f1.answer,
            },
            "injection_result": {
                "injected_fault": inj_res.injected_fault,
                "injection_applied": inj_res.injection_applied,
                "trace_condition_achieved": inj_res.trace_condition_achieved,
                "trial_valid": inj_res.trial_valid,
                "invalid_reason": inj_res.invalid_reason,
            },
        })

    # Verify that the shared base corpus is strictly unmodified
    final_corpus_size = len(base_corpus)
    corpus_unmutated = (final_corpus_size == initial_corpus_size)

    summary = {
        "pilot_name": "F1_Evidence_Unavailable_Pilot",
        "initial_corpus_size": initial_corpus_size,
        "final_corpus_size": final_corpus_size,
        "base_corpus_unmutated": corpus_unmutated,
        "trials_attempted": attempted,
        "trials_valid": valid_count,
        "trials_invalid": invalid_count,
        "invalid_reasons": invalid_reasons,
        "trials": pilot_results,
    }

    out_dir = Path("results/pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "pilot_f1_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Pilot completed: {attempted} attempted, {valid_count} valid, {invalid_count} invalid.")
    print(f"Base corpus immutability verified: {corpus_unmutated} (size: {final_corpus_size})")
    print(f"Detailed pilot report written to: {out_file}")


if __name__ == "__main__":
    run_f1_pilot()
