"""Unit tests for D4: Learned Classifier Attribution Policy."""

import numpy as np
from rag_autopsy.core.schemas import DiagnosticTrace, RetrievedPassage
from rag_autopsy.diagnostics.d4_learned import D4LearnedClassifierPolicy


def test_d4_learned_feature_extraction():
    p1 = RetrievedPassage(passage_id="p1", rank=1, score=12.0, text="First passage text.")
    p2 = RetrievedPassage(passage_id="p2", rank=2, score=8.0, text="Second passage text.")

    trace = DiagnosticTrace(
        trial_id="trace_learn_001",
        question_id="q1",
        question="What is this?",
        retrieved_top_k=[p1, p2],
        source_passage_ids=["p1", "p2"],
        candidate_pool_scores=[12.0, 8.0, 5.0, 2.0, 1.0],
        pre_assembly_passage_ids=["p1", "p2"],
        post_assembly_passage_ids=["p1"],  # 1 dropped
        generator_context="[Passage 1]: First passage text.",
        answer="First passage text content.",
        latency_ms=10.0,
    )

    feats = D4LearnedClassifierPolicy.extract_features(trace)
    assert len(feats) == len(D4LearnedClassifierPolicy.FEATURE_NAMES)
    assert feats[0] == 1.0  # 1 drop
    assert feats[1] == 0.5  # survival ratio 1/2


def test_d4_learned_fit_and_diagnose():
    p1 = RetrievedPassage(passage_id="p1", rank=1, score=10.0, text="Clean passage.")

    t_f0 = DiagnosticTrace(
        trial_id="tr_f0",
        question_id="q1",
        question="clean question",
        retrieved_top_k=[p1],
        source_passage_ids=["p1"],
        pre_assembly_passage_ids=["p1"],
        post_assembly_passage_ids=["p1"],
        generator_context="Clean passage.",
        answer="Clean passage.",
        latency_ms=5.0,
    )

    t_f3 = DiagnosticTrace(
        trial_id="tr_f3",
        question_id="q2",
        question="dropped question",
        retrieved_top_k=[p1],
        source_passage_ids=["p1"],
        pre_assembly_passage_ids=["p1"],
        post_assembly_passage_ids=[],  # dropped
        generator_context="",
        answer="I do not know.",
        latency_ms=5.0,
    )

    policy = D4LearnedClassifierPolicy(classifier_type="decision_tree")
    policy.fit([t_f0, t_f3], ["F0", "F3"])
    assert policy.is_fitted is True

    # Diagnose new trace with a drop
    diag_f3 = policy.diagnose(t_f3)
    assert diag_f3.predicted_cause == "F3"
    assert diag_f3.confidence > 0.5

    # Diagnose clean trace
    diag_f0 = policy.diagnose(t_f0)
    assert diag_f0.predicted_cause == "F0"
    assert diag_f0.confidence > 0.5
