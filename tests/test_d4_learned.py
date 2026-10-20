"""Unit tests for D4: Learned Classifier Attribution Policy."""

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
        trial_id="trial_learn_001",
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
        trial_id="trial_learn_002",
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


def test_d4_learned_avoids_train_on_test_leakage():
    """Verify that runner fits D4 learned policy on dev traces, not test traces."""
    from rag_autopsy.data.schema import QARecord
    from rag_autopsy.data.splits import SplitManifest
    from rag_autopsy.evaluation.runner import run_experiment_harness

    # Create dummy records across dev and test
    dev_rec = QARecord(
        question_id="q_dev_1",
        question="Dev question?",
        gold_answer="Dev answer",
        question_type="bridge",
        difficulty_level="easy",
        gold_evidence_titles=["DevTitle"],
        gold_evidence_passage_ids=["dev_p1"],
        passages=[],
    )
    test_rec = QARecord(
        question_id="q_test_1",
        question="Test question?",
        gold_answer="Test answer",
        question_type="bridge",
        difficulty_level="easy",
        gold_evidence_titles=["TestTitle"],
        gold_evidence_passage_ids=["test_p1"],
        passages=[],
    )

    manifest = SplitManifest(
        split_version="1.0.0",
        total_questions=2,
        dev_ids=["q_dev_1"],
        calibration_ids=[],
        test_ids=["q_test_1"],
        random_seed=42,
        split_counts={"dev": 1, "calibration": 0, "test": 1},
    )

    results = run_experiment_harness(
        qa_records=[dev_rec, test_rec],
        split_manifest=manifest,
        target_split="test",
        include_d4_learned=True,
        n_bootstraps=5,
        seed=42,
    )

    assert "d4_learned_decision_tree" in results["diagnostic_results"]
    d4_metrics = results["diagnostic_results"]["d4_learned_decision_tree"]["metrics"]
    assert d4_metrics["total_trials"] > 0
    # Evaluated trials belong to test split
    test_trial_rows = [r for r in results["trial_rows"] if r["question_id"] == "q_test_1"]
    assert len(test_trial_rows) > 0

