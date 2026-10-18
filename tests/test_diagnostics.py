"""Unit and behavioral tests for diagnostic methods and attribution policies."""

import pytest
from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord, RetrievedPassage
from rag_autopsy.diagnostics.baselines import MajorityCauseBaseline, RandomBaseline
from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic
from rag_autopsy.diagnostics.d2_support import D2LexicalSupportDiagnostic
from rag_autopsy.diagnostics.d4_policy import D4CombinedAttributionPolicy


def test_base_diagnostic_type_safety(
    sample_diagnostic_trace: DiagnosticTrace, sample_evaluator_record: EvaluatorRecord
) -> None:
    diag = D1RetrievalDiagnostic()

    # Valid invocation with DiagnosticTrace
    res = diag.diagnose(sample_diagnostic_trace)
    assert res.predicted_cause in ["F0", "F1", "F2", "F3"]

    # Illegal invocation with privileged EvaluatorRecord
    with pytest.raises(TypeError, match="Data boundary violation"):
        diag.diagnose(sample_evaluator_record)  # type: ignore[arg-type]


def test_majority_cause_baseline(sample_diagnostic_trace: DiagnosticTrace) -> None:
    baseline = MajorityCauseBaseline()
    baseline.fit(["F1", "F1", "F2", "F0"])
    assert baseline.majority_cause == "F1"

    diag = baseline.diagnose(sample_diagnostic_trace)
    assert diag.predicted_cause == "F1"


def test_random_baseline(sample_diagnostic_trace: DiagnosticTrace) -> None:
    baseline = RandomBaseline(seed=42)
    diag = baseline.diagnose(sample_diagnostic_trace)
    assert diag.predicted_cause in ["F0", "F1", "F2", "F3", "F4"]


def test_d1_retrieval_f3_structural_detection() -> None:
    diag = D1RetrievalDiagnostic()
    trace = DiagnosticTrace(
        trial_id="trace_test_01",
        question_id="q01",
        question="What is X?",
        retrieved_top_k=[
            RetrievedPassage(passage_id="p1", rank=1, score=10.0, text="Text 1"),
            RetrievedPassage(passage_id="p2", rank=2, score=8.0, text="Text 2"),
        ],
        pre_assembly_passage_ids=["p1", "p2"],
        post_assembly_passage_ids=["p1"],  # p2 was omitted
        generator_context="Text 1",
        answer="X is 1",
        latency_ms=5.0,
    )
    res = diag.diagnose(trace)
    assert res.predicted_cause == "F3"
    assert "Structural context loss" in res.evidence_rationale


def test_d1_regime_a_vs_regime_b() -> None:
    # Low score trace
    low_score_trace = DiagnosticTrace(
        trial_id="trace_test_02",
        question_id="q02",
        question="What is Y?",
        retrieved_top_k=[
            RetrievedPassage(passage_id="p1", rank=1, score=0.5, text="Irrelevant distractor"),
        ],
        candidate_pool_scores=[0.5, 0.4, 0.3, 0.2, 0.1],
        pre_assembly_passage_ids=["p1"],
        post_assembly_passage_ids=["p1"],
        generator_context="Irrelevant distractor",
        answer="I do not know",
        latency_ms=5.0,
    )

    # In Regime A: Predicts F1 due to score depression
    diag_a = D1RetrievalDiagnostic(regime="regime_a", low_score_threshold=2.0)
    res_a = diag_a.diagnose(low_score_trace)
    assert res_a.predicted_cause == "F1"

    # In Regime B: Abstains with UNKNOWN due to inherent F1/F2 ambiguity
    diag_b = D1RetrievalDiagnostic(regime="regime_b", abstain_on_ambiguity=True)
    res_b = diag_b.diagnose(low_score_trace)
    assert res_b.abstained is True
    assert res_b.predicted_cause == "UNKNOWN"
    assert "AMBIGUOUS_RETRIEVAL_MISS" in str(res_b.abstention_reason)


def test_d2_contradiction_detection() -> None:
    diag = D2LexicalSupportDiagnostic()
    trace = DiagnosticTrace(
        trial_id="trace_test_03",
        question_id="q03",
        question="What is the speed limit?",
        retrieved_top_k=[
            RetrievedPassage(passage_id="p1", rank=1, score=10.0, text="Official records contradict earlier claims: the speed limit is not 50."),
        ],
        pre_assembly_passage_ids=["p1"],
        post_assembly_passage_ids=["p1"],
        generator_context="Official records contradict earlier claims: the speed limit is not 50.",
        answer="The limit is 50",
        latency_ms=5.0,
    )
    res = diag.diagnose(trace)
    assert res.predicted_cause == "F4"
    assert "contradiction" in res.evidence_rationale.lower()


def test_d4_combined_policy_precedence() -> None:
    policy = D4CombinedAttributionPolicy()

    # Case 1: Assembly Loss (F3) takes priority
    trace_f3 = DiagnosticTrace(
        trial_id="trace_test_04",
        question_id="q04",
        question="What is Z?",
        retrieved_top_k=[
            RetrievedPassage(passage_id="p1", rank=1, score=10.0, text="Z is 100."),
            RetrievedPassage(passage_id="p2", rank=2, score=8.0, text="Z is vital."),
        ],
        pre_assembly_passage_ids=["p1", "p2"],
        post_assembly_passage_ids=["p1"],  # p2 lost
        generator_context="Z is 100.",
        answer="Z is 100",
        latency_ms=5.0,
    )
    res_f3 = policy.diagnose(trace_f3)
    assert res_f3.predicted_cause == "F3"

    # Case 2: Clean execution (F0) when retrieval and support agree
    trace_f0 = DiagnosticTrace(
        trial_id="trace_test_05",
        question_id="q05",
        question="What is the capital of Germany?",
        retrieved_top_k=[
            RetrievedPassage(passage_id="p1", rank=1, score=12.0, text="Berlin is the capital of Germany."),
        ],
        candidate_pool_scores=[12.0, 9.0, 7.0, 5.0, 3.0],
        pre_assembly_passage_ids=["p1"],
        post_assembly_passage_ids=["p1"],
        generator_context="Berlin is the capital of Germany.",
        answer="Berlin",
        latency_ms=5.0,
    )
    res_f0 = policy.diagnose(trace_f0)
    assert res_f0.predicted_cause == "F0"
    assert res_f0.abstained is False


def test_d1_misdiagnosis_failure_case() -> None:
    """Documented blind spot for D1: Naturally low-scoring query misdiagnosed as F1.
    
    A clean trial (F0) with low BM25 score (e.g. domain vocabulary with low inverse document frequency)
    triggers D1's low_score_threshold (< 2.0), causing D1 to falsely attribute F1 (Evidence Unavailable).
    """
    diag = D1RetrievalDiagnostic(regime="regime_a", low_score_threshold=2.0)
    clean_trace_low_score = DiagnosticTrace(
        trial_id="trace_blindspot_d1",
        question_id="q_rare",
        question="What is it?",
        retrieved_top_k=[
            RetrievedPassage(passage_id="p_gold_clean", rank=1, score=1.2, text="It is a specialized device."),
        ],
        candidate_pool_scores=[1.2, 0.9, 0.5],
        pre_assembly_passage_ids=["p_gold_clean"],
        post_assembly_passage_ids=["p_gold_clean"],
        generator_context="It is a specialized device.",
        answer="A specialized device",
        latency_ms=5.0,
    )
    # The true state was F0 (clean, evidence retrieved), but D1 misdiagnoses F1!
    res = diag.diagnose(clean_trace_low_score)
    assert res.predicted_cause == "F1"  # Documents the exact failure mode of D1


def test_d2_misdiagnosis_failure_case_paraphrase() -> None:
    """Documented blind spot for D2: Valid abstractive paraphrase misdiagnosed as F1.
    
    A clean trial (F0) where the generator answers using synonyms and abstraction
    (e.g., 'physician' for 'medical doctor', 'perished' for 'died').
    Because lexical keyword overlap falls below min_overlap_ratio (0.3), D2 falsely diagnoses F1.
    """
    diag = D2LexicalSupportDiagnostic(min_overlap_ratio=0.3)
    clean_trace_paraphrase = DiagnosticTrace(
        trial_id="trace_blindspot_d2_para",
        question_id="q_synonym",
        question="What happened to the explorer?",
        retrieved_top_k=[
            RetrievedPassage(passage_id="p1", rank=1, score=10.0, text="The medical doctor died during the expedition."),
        ],
        pre_assembly_passage_ids=["p1"],
        post_assembly_passage_ids=["p1"],
        generator_context="The medical doctor died during the expedition.",
        answer="The physician perished while traveling abroad.",  # 0% keyword overlap with context!
        latency_ms=5.0,
    )
    # The answer is factually correct and supported, but D2's lexical check fails and predicts F1!
    res = diag.diagnose(clean_trace_paraphrase)
    assert res.predicted_cause == "F1"  # Documents D2 lexical blind spot


def test_d2_misdiagnosis_failure_case_hallucination_entity_copy() -> None:
    """Documented blind spot for D2: Factually false hallucination copying context entities misdiagnosed as F0.
    
    A hallucinated response (F5) that repeats the subject entity but fabricates the predicate
    (e.g., 'Albert Einstein measured the speed of light in 1905').
    Because words like 'Albert', 'Einstein', 'speed', 'light' match the context, overlap >= 0.3
    and D2 falsely diagnoses F0 (Grounded).
    """
    diag = D2LexicalSupportDiagnostic(min_overlap_ratio=0.3)
    hallucinated_trace = DiagnosticTrace(
        trial_id="trace_blindspot_d2_halluc",
        question_id="q_halluc",
        question="Who measured the speed of light?",
        retrieved_top_k=[
            RetrievedPassage(
                passage_id="p1",
                rank=1,
                score=10.0,
                text="The speed of light in vacuum is defined as 299,792,458 metres per second.",
            ),
        ],
        pre_assembly_passage_ids=["p1"],
        post_assembly_passage_ids=["p1"],
        generator_context="The speed of light in vacuum is defined as 299,792,458 metres per second.",
        answer="The speed of light was measured by Albert Einstein in vacuum.",
        latency_ms=5.0,
    )
    # Factual hallucination (Einstein didn't measure it), but lexical overlap >= 0.3 tricks D2 into predicting F0!
    res = diag.diagnose(hallucinated_trace)
    assert res.predicted_cause == "F0"  # Documents D2 false grounding blind spot


def test_diagnostics_strict_evaluator_leakage_rejection() -> None:
    """Verify that neither D1 nor D2 accepts EvaluatorRecord or forbidden evaluator keys."""
    d1 = D1RetrievalDiagnostic()
    d2 = D2LexicalSupportDiagnostic()

    eval_rec = EvaluatorRecord(
        trial_id="trial_leak_test_01",
        question_id="q01",
        split="test",
        injected_fault="F1",
        gold_answers=["test"],
        gold_evidence_ids=["p1"],
        trial_valid=True,
    )

    with pytest.raises(TypeError, match="Data boundary violation"):
        d1.diagnose(eval_rec)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Data boundary violation"):
        d2.diagnose(eval_rec)  # type: ignore[arg-type]
