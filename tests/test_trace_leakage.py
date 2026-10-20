"""Tests ensuring strict structural isolation between visible traces and privileged evaluator data."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_autopsy.core.schemas import (
    Diagnosis,
    DiagnosticTrace,
    EvaluatorRecord,
    EvaluatorStore,
    RetrievedPassage,
    explicit_evaluation_join,
    generate_stable_trial_id,
    load_evaluator_records_jsonl,
    load_traces_jsonl,
    save_evaluator_records_jsonl,
    save_traces_jsonl,
)
from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic
from rag_autopsy.diagnostics.d3_nli import D3NLISupportDiagnostic
from rag_autopsy.diagnostics.d4_learned import D4LearnedClassifierPolicy
from rag_autopsy.pipeline.generator import CachedGenerator, MockExtractiveGenerator


def test_diagnostic_trace_has_no_evaluator_fields() -> None:
    """Verify that DiagnosticTrace model schema does not define privileged fields."""
    forbidden_fields = {
        "injected_fault",
        "fault_parameters",
        "gold_answer",
        "gold_answers",
        "gold_evidence_ids",
        "target_rank",
        "injection_applied",
        "trial_valid",
        "invalid_reason",
    }
    model_fields = set(DiagnosticTrace.model_fields.keys())
    intersection = model_fields.intersection(forbidden_fields)
    assert not intersection, f"DiagnosticTrace schema leaks privileged fields: {intersection}"


def test_extra_fields_forbidden_in_diagnostic_trace() -> None:
    """Ensure extra='forbid' rejects arbitrary or smuggled fields in DiagnosticTrace."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DiagnosticTrace(
            trial_id="trial_999999",
            question_id="q001",
            question="What is X?",
            retrieved_top_k=[],
            pre_assembly_passage_ids=[],
            post_assembly_passage_ids=[],
            generator_context="",
            answer="",
            latency_ms=1.0,
            injected_fault="F1",  # Forbidden extra field!
        )


def test_extra_fields_forbidden_in_retrieved_passage() -> None:
    """Ensure extra='forbid' rejects privileged fields like is_gold in RetrievedPassage."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RetrievedPassage(
            passage_id="p1",
            rank=1,
            score=10.0,
            text="Context",
            is_gold_evidence=True,  # Forbidden extra field!
        )


def test_leakage_rejection_in_metadata_key() -> None:
    """Ensure that attempts to smuggle privileged fields inside metadata dict raise a ValueError."""
    passages = [
        RetrievedPassage(passage_id="p1", rank=1, score=5.0, text="Sample text")
    ]
    with pytest.raises(ValidationError, match="Leakage violation"):
        DiagnosticTrace(
            trial_id="trial_3a7b9c1d0e4f826a",
            question_id="q001",
            question="What is X?",
            retrieved_top_k=passages,
            pre_assembly_passage_ids=["p1"],
            post_assembly_passage_ids=["p1"],
            generator_context="Sample text",
            answer="X is 1",
            latency_ms=10.0,
            metadata={"injected_fault": "F1"},
        )


def test_leakage_rejection_in_metadata_value() -> None:
    """Ensure that attempts to smuggle fault tokens inside metadata string values raise a ValueError."""
    passages = [
        RetrievedPassage(passage_id="p1", rank=1, score=5.0, text="Sample text")
    ]
    with pytest.raises(ValidationError, match="Leakage violation"):
        DiagnosticTrace(
            trial_id="trial_3a7b9c1d0e4f826a",
            question_id="q001",
            question="What is X?",
            retrieved_top_k=passages,
            pre_assembly_passage_ids=["p1"],
            post_assembly_passage_ids=["p1"],
            generator_context="Sample text",
            answer="X is 1",
            latency_ms=10.0,
            metadata={"experiment_type": "f1_injection"},
        )


def test_metadata_evaluator_object_smuggling_rejected(
    sample_evaluator_record: EvaluatorRecord,
) -> None:
    """Ensure that placing an EvaluatorRecord inside metadata raises a ValueError."""
    with pytest.raises(ValidationError, match="evaluator object smuggled"):
        DiagnosticTrace(
            trial_id="trial_3a7b9c1d0e4f826a",
            question_id="q001",
            question="What is X?",
            retrieved_top_k=[],
            pre_assembly_passage_ids=[],
            post_assembly_passage_ids=[],
            generator_context="",
            answer="",
            latency_ms=1.0,
            metadata={"eval_ref": sample_evaluator_record},
        )


def test_leakage_rejection_in_passage_id_and_title() -> None:
    """Ensure that attempts to include fault tokens in passage IDs or titles raise ValueError."""
    bad_passage_id = RetrievedPassage(
        passage_id="q001__f2_distractor_1",
        rank=1,
        score=5.0,
        text="Sample text",
        title="Valid Title",
    )
    with pytest.raises(ValidationError, match="Leakage violation"):
        DiagnosticTrace(
            trial_id="trial_3a7b9c1d0e4f826a",
            question_id="q001",
            question="What is X?",
            retrieved_top_k=[bad_passage_id],
            pre_assembly_passage_ids=["q001__f2_distractor_1"],
            post_assembly_passage_ids=["q001__f2_distractor_1"],
            generator_context="Sample text",
            answer="X is 1",
            latency_ms=10.0,
        )

    bad_passage_title = RetrievedPassage(
        passage_id="passage_syn_12345678",
        rank=1,
        score=5.0,
        text="Sample text",
        title="Distractor Overview regarding X",
    )
    with pytest.raises(ValidationError, match="Leakage violation"):
        DiagnosticTrace(
            trial_id="trial_3a7b9c1d0e4f826a",
            question_id="q001",
            question="What is X?",
            retrieved_top_k=[bad_passage_title],
            pre_assembly_passage_ids=["passage_syn_12345678"],
            post_assembly_passage_ids=["passage_syn_12345678"],
            generator_context="Sample text",
            answer="X is 1",
            latency_ms=10.0,
        )


@pytest.mark.parametrize(
    "bad_id",
    [
        "trial-f1-severe",
        "trial_f2_step",
        "trial_clean_control",
        "trial_hallucination",
        "f3_trial",
        "trial_ranked_out",
    ],
)
def test_leakage_rejection_in_trial_id(bad_id: str) -> None:
    """Ensure that encoding fault names or tokens in trial_id raises a ValidationError."""
    with pytest.raises(ValidationError, match="Leakage violation"):
        DiagnosticTrace(
            trial_id=bad_id,
            question_id="q001",
            question="What is X?",
            retrieved_top_k=[],
            pre_assembly_passage_ids=[],
            post_assembly_passage_ids=[],
            generator_context="",
            answer="X",
            latency_ms=10.0,
        )


def test_stable_non_label_encoding_ids() -> None:
    """Verify that generate_stable_trial_id produces opaque hexadecimal IDs with no fault tokens."""
    tid_0 = generate_stable_trial_id(seed=42, question_id="q001", intervention_key="F0")
    tid_1 = generate_stable_trial_id(seed=42, question_id="q001", intervention_key="F1")
    tid_2 = generate_stable_trial_id(seed=42, question_id="q001", intervention_key="F2")

    assert tid_0.startswith("trial_")
    assert len(tid_0) == 22  # "trial_" + 16 hex chars
    assert tid_0 != tid_1 != tid_2

    # None of the IDs contain isolated fault tokens
    for tid in [tid_0, tid_1, tid_2]:
        hex_part = tid.replace("trial_", "")
        assert not any(f"_{f}_" in tid for f in ["f0", "f1", "f2", "f3", "f4", "f5"])


def test_file_boundary_separate_jsonl_serialization(
    tmp_path: Path,
    sample_diagnostic_trace: DiagnosticTrace,
    sample_evaluator_record: EvaluatorRecord,
) -> None:
    """Verify that traces and evaluator records are stored in separate files and cross-loading fails."""
    traces_file = tmp_path / "diagnostic_traces.jsonl"
    eval_file = tmp_path / "evaluator_records.jsonl"

    save_traces_jsonl([sample_diagnostic_trace], traces_file)
    save_evaluator_records_jsonl([sample_evaluator_record], eval_file)

    assert traces_file.is_file()
    assert eval_file.is_file()

    # Successful loading from correct files
    loaded_traces = load_traces_jsonl(traces_file)
    assert len(loaded_traces) == 1
    assert loaded_traces[0].trial_id == sample_diagnostic_trace.trial_id

    loaded_eval = load_evaluator_records_jsonl(eval_file)
    assert len(loaded_eval) == 1
    assert loaded_eval[0].trial_id == sample_evaluator_record.trial_id

    # Cross-loading MUST fail due to extra='forbid'
    with pytest.raises(ValidationError):
        load_traces_jsonl(eval_file)

    with pytest.raises(ValidationError):
        load_evaluator_records_jsonl(traces_file)


def test_evaluator_store_encapsulation(sample_evaluator_record: EvaluatorRecord) -> None:
    """Verify EvaluatorStore strictly encapsulates records and rejects duplicates or invalid types."""
    store = EvaluatorStore()
    store.add(sample_evaluator_record)
    assert len(store) == 1
    assert sample_evaluator_record.trial_id in store
    assert store.get(sample_evaluator_record.trial_id) == sample_evaluator_record

    # Duplicate trial_id rejected
    with pytest.raises(ValueError, match="Duplicate trial_id"):
        store.add(sample_evaluator_record)

    # Invalid record type rejected
    with pytest.raises(TypeError):
        store.add({"trial_id": "bad"})  # type: ignore[arg-type]


def test_diagnostic_api_strictly_rejects_non_trace_types(
    sample_evaluator_record: EvaluatorRecord,
) -> None:
    """Ensure BaseDiagnostic.diagnose strictly rejects anything that is not a DiagnosticTrace."""
    diag = D1RetrievalDiagnostic()

    # Reject EvaluatorRecord
    with pytest.raises(TypeError, match="Data boundary violation"):
        diag.diagnose(sample_evaluator_record)  # type: ignore[arg-type]

    # Reject dict
    with pytest.raises(TypeError, match="Data boundary violation"):
        diag.diagnose({"trial_id": "t1"})  # type: ignore[arg-type]


def test_diagnostic_api_rejects_tampered_trace(sample_diagnostic_trace: DiagnosticTrace) -> None:
    """Ensure BaseDiagnostic.diagnose catches dynamically injected forbidden attributes."""
    diag = D1RetrievalDiagnostic()
    # Dynamically inject attribute bypassing pydantic init
    object.__setattr__(sample_diagnostic_trace, "injected_fault", "F1")

    with pytest.raises(ValueError, match="Data boundary violation"):
        diag.diagnose(sample_diagnostic_trace)


def test_feature_construction_isolation(sample_diagnostic_trace: DiagnosticTrace) -> None:
    """Verify that learned policy feature extraction only accesses DiagnosticTrace fields."""
    feats = D4LearnedClassifierPolicy.extract_features(sample_diagnostic_trace)
    assert len(feats) == 8
    assert all(isinstance(v, (float, int)) for v in feats)


def test_cached_generator_keys_contain_no_evaluator_data(tmp_path: Path) -> None:
    """Verify that CachedGenerator key and disk cache payload contain zero evaluator data."""
    inner = MockExtractiveGenerator()
    cached_gen = CachedGenerator(inner_generator=inner, cache_dir=str(tmp_path))

    q = "What is the capital of France?"
    ctx = "Paris is the capital of France."
    key = cached_gen.compute_cache_key(q, ctx)
    ans = cached_gen.generate(q, ctx)
    assert "Paris" in ans

    cache_file = tmp_path / f"{key}.json"
    assert cache_file.is_file()

    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure no evaluator keys in cached file
    assert "injected_fault" not in data
    assert "gold_answer" not in data
    assert "gold_evidence_ids" not in data
    assert "target_rank" not in data


def test_d3_nli_cache_keys_contain_no_evaluator_data(tmp_path: Path) -> None:
    """Verify that D3 NLI cache computes keys strictly on premise and hypothesis strings."""
    nli = D3NLISupportDiagnostic(cache_dir=str(tmp_path), load_model_weights=False)
    premise = "Prague is on the Vltava River."
    hypothesis = "The city is near a river."

    key = nli._compute_cache_key(premise, hypothesis)
    assert len(key) == 64  # SHA-256 hex string

    # Key should only change if premise or hypothesis changes
    key_identical = nli._compute_cache_key(premise, hypothesis)
    assert key == key_identical


def test_explicit_evaluation_join_integrity(
    sample_diagnostic_trace: DiagnosticTrace,
    sample_evaluator_record: EvaluatorRecord,
) -> None:
    """Verify explicit_evaluation_join enforces 1:1 join integrity on trial_id."""
    store = EvaluatorStore()
    store.add(sample_evaluator_record)

    diagnosis = Diagnosis(
        trial_id=sample_diagnostic_trace.trial_id,
        diagnostic_name="d1_test",
        predicted_cause="F0",
        confidence=0.9,
        abstained=False,
        evidence_rationale="Test rationale",
        runtime_ms=1.0,
    )

    joined = explicit_evaluation_join(
        traces=[sample_diagnostic_trace],
        diagnoses_by_diagnostic={"d1_test": [diagnosis]},
        evaluator_store=store,
    )

    assert len(joined) == 1
    assert joined[0].trial_id == sample_diagnostic_trace.trial_id
    assert joined[0].trace == sample_diagnostic_trace
    assert joined[0].evaluator_record == sample_evaluator_record
    assert joined[0].diagnoses["d1_test"] == diagnosis

    # Mismatched trial_id raises KeyError
    bad_trace = sample_diagnostic_trace.model_copy(update={"trial_id": "trial_missing_999"})
    with pytest.raises(KeyError, match="Join integrity failure"):
        explicit_evaluation_join(
            traces=[bad_trace],
            diagnoses_by_diagnostic={"d1_test": [diagnosis]},
            evaluator_store=store,
        )


def test_sort_order_decoupling_prevents_modulo_attack(
    synthetic_fixture_path: Path,
) -> None:
    """Verify that runner presentation order is decoupled from fault type.
    
    Proves that an adversarial diagnoser assuming sequential [F0, F1, F2, F3, F4] order
    cannot reliably predict the fault label from list index.
    """
    from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
    from rag_autopsy.data.splits import create_question_splits
    from rag_autopsy.evaluation.runner import run_experiment_harness

    raw = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw)
    manifest = create_question_splits(records, dev_ratio=1.0, calibration_ratio=0.0, test_ratio=0.0, seed=42)

    res = run_experiment_harness(
        qa_records=records,
        split_manifest=manifest,
        target_split="dev",
        seed=42,
    )

    traces = res["traces"]
    eval_store = res["evaluator_store"]

    assert len(traces) == 10  # 2 questions * 5 faults

    # Check the actual sequence of faults in traces
    actual_sequence = [eval_store.get(t.trial_id).injected_fault for t in traces]

    # Sequential order would be F0, F1, F2, F3, F4, F0, F1, F2, F3, F4
    sequential_order = ["F0", "F1", "F2", "F3", "F4"] * 2
    assert actual_sequence != sequential_order, "Presentation order must NOT match predictable sequential fault order!"

    # Adversarial modulo predictor: y_hat = ["F0", "F1", "F2", "F3", "F4"][i % 5]
    adversary_preds = [["F0", "F1", "F2", "F3", "F4"][i % 5] for i in range(len(traces))]
    correct_matches = sum(p == a for p, a in zip(adversary_preds, actual_sequence))
    adversary_accuracy = correct_matches / len(traces)

    # Adversary achieves no better than chance (~0.20), not 1.0
    assert adversary_accuracy <= 0.40, f"Adversary exploited order with inflated accuracy: {adversary_accuracy}"


def test_train_calibration_test_manifest_strictly_grouped_by_question(
    synthetic_fixture_path: Path,
) -> None:
    """Verify that split manifests strictly group by question_id and contain no fault labels."""
    from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
    from rag_autopsy.data.splits import create_question_splits

    raw = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw)
    manifest = create_question_splits(records, dev_ratio=0.5, calibration_ratio=0.0, test_ratio=0.5, seed=42)

    # 1. 0% split overlap
    assert len(set(manifest.dev_ids).intersection(set(manifest.test_ids))) == 0
    assert len(set(manifest.dev_ids).intersection(set(manifest.calibration_ids))) == 0

    # 2. Manifest maps question_id only
    assert all(qid.startswith("syn_q") for qid in manifest.dev_ids)
    assert all(qid.startswith("syn_q") for qid in manifest.test_ids)

    # 3. Manifest contains no fault labels or trial IDs
    manifest_dict = manifest.model_dump()
    manifest_str = json.dumps(manifest_dict).lower()
    for forbidden in ["f0", "f1", "f2", "f3", "f4", "f5", "trial_"]:
        assert f'"{forbidden}"' not in manifest_str





def test_logging_does_not_smuggle_privileged_evaluator_data(
    caplog: pytest.LogCaptureFixture,
    sample_diagnostic_trace: DiagnosticTrace,
) -> None:
    """Verify that diagnostic logging does not output forbidden evaluator keys."""
    import logging

    from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic

    logger = logging.getLogger("rag_autopsy")
    logger.setLevel(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        diag = D1RetrievalDiagnostic()
        diagnosis = diag.diagnose(sample_diagnostic_trace)
        logger.info(f"Diagnostic output: {diagnosis.model_dump_json()}")

    log_text = caplog.text.lower()
    assert "gold_answer" not in log_text
    assert "gold_evidence_ids" not in log_text
    assert "target_rank" not in log_text


