"""Unit tests for RAG quality baselines (Context Precision and Faithfulness)."""

from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord, RetrievedPassage
from rag_autopsy.evaluation.rag_quality import (
    compute_context_precision_at_k,
    compute_faithfulness,
    evaluate_rag_quality_dataset,
    split_into_claims,
)


def test_context_precision_at_k_exact_scenarios() -> None:
    p1 = RetrievedPassage(passage_id="gold_1", rank=1, score=10.0, text="Gold passage 1")
    p2 = RetrievedPassage(passage_id="distractor_1", rank=2, score=8.0, text="Distractor 1")
    p3 = RetrievedPassage(passage_id="gold_2", rank=3, score=6.0, text="Gold passage 2")

    # 1. Gold at rank 1 only
    cp_1 = compute_context_precision_at_k([p1, p2], gold_evidence_ids=["gold_1"])
    assert cp_1 == 1.0

    # 2. Gold at rank 2 only
    cp_2 = compute_context_precision_at_k([p2, p1], gold_evidence_ids=["gold_1"])
    assert cp_2 == 0.5  # Precision@2 = 1/2

    # 3. Gold at rank 1 and 3: (1/1 + 2/3) / 2 = (1 + 0.6667)/2 = 0.8333
    cp_3 = compute_context_precision_at_k([p1, p2, p3], gold_evidence_ids=["gold_1", "gold_2"])
    assert round(cp_3, 3) == 0.833

    # 4. Gold not in retrieved list
    cp_none = compute_context_precision_at_k([p2], gold_evidence_ids=["gold_1"])
    assert cp_none == 0.0

    # 5. Empty inputs
    assert compute_context_precision_at_k([], ["gold_1"]) == 0.0
    assert compute_context_precision_at_k([p1], []) == 0.0


def test_faithfulness_scenarios() -> None:
    context = "The Eiffel Tower is located in Paris, France. It was constructed from 1887 to 1889."

    # 1. Fully grounded claim
    assert compute_faithfulness("The Eiffel Tower is in Paris, France.", context) == 1.0

    # 2. Completely hallucinated claim
    assert compute_faithfulness("The Eiffel Tower is in Berlin, Germany.", context) == 0.0

    # 3. Two claims: one grounded, one ungrounded
    answer_mixed = "The Eiffel Tower is in Paris, France. It was designed by Leonardo da Vinci."
    faith_mixed = compute_faithfulness(answer_mixed, context)
    assert 0.4 <= faith_mixed <= 0.6

    # 4. Empty context or answer
    assert compute_faithfulness("", context) == 1.0
    assert compute_faithfulness("Paris is the capital.", "") == 0.0


def test_evaluate_rag_quality_dataset_comparison() -> None:
    p_gold = RetrievedPassage(passage_id="p_gold", rank=1, score=9.0, text="Berlin is the capital of Germany.")
    p_dist = RetrievedPassage(passage_id="p_dist", rank=1, score=9.0, text="Munich is in Bavaria.")

    # F0 Clean trial
    tr_0 = DiagnosticTrace(
        trial_id="trial_1111111111111111",
        question_id="q1",
        question="What is the capital of Germany?",
        retrieved_top_k=[p_gold],
        pre_assembly_passage_ids=["p_gold"],
        post_assembly_passage_ids=["p_gold"],
        generator_context="Berlin is the capital of Germany.",
        answer="Berlin",
        latency_ms=10.0,
    )
    ev_0 = EvaluatorRecord(
        trial_id="trial_1111111111111111",
        question_id="q1",
        split="test",
        injected_fault="F0",
        gold_answers=["Berlin"],
        gold_evidence_ids=["p_gold"],
        trial_valid=True,
    )

    # F1 Unavailable trial
    tr_1 = DiagnosticTrace(
        trial_id="trial_2222222222222222",
        question_id="q1",
        question="What is the capital of Germany?",
        retrieved_top_k=[p_dist],
        pre_assembly_passage_ids=["p_dist"],
        post_assembly_passage_ids=["p_dist"],
        generator_context="Munich is in Bavaria.",
        answer="Munich",
        latency_ms=10.0,
    )
    ev_1 = EvaluatorRecord(
        trial_id="trial_2222222222222222",
        question_id="q1",
        split="test",
        injected_fault="F1",
        gold_answers=["Berlin"],
        gold_evidence_ids=["p_gold"],
        trial_valid=True,
    )

    res = evaluate_rag_quality_dataset([tr_0, tr_1], [ev_0, ev_1])

    assert res.total_trials == 2
    assert "F0" in res.profiles_by_fault
    assert "F1" in res.profiles_by_fault
    assert res.profiles_by_fault["F0"].mean_context_precision == 1.0
    assert res.profiles_by_fault["F1"].mean_context_precision == 0.0
    assert "conflate distinct failure modes" in res.interpretability_gap_summary
