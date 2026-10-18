"""Unit tests for answer correctness metrics and text normalization."""

from rag_autopsy.evaluation.answer_metrics import (
    compute_exact_match,
    compute_token_f1,
    evaluate_answer_correctness,
    normalize_answer,
)


def test_normalize_answer() -> None:
    assert normalize_answer("The Vltava River!") == "vltava river"
    assert normalize_answer("  An   apple  ") == "apple"
    assert normalize_answer("a cat, a dog.") == "cat dog"
    assert normalize_answer("") == ""


def test_compute_exact_match() -> None:
    assert compute_exact_match("Vltava River", "the vltava river.") is True
    assert compute_exact_match("Berlin", "berlin") is True
    assert compute_exact_match("Paris", "London") is False
    assert compute_exact_match("Kafka", "") is False


def test_compute_token_f1() -> None:
    # Exact match gives F1 = 1.0
    assert compute_token_f1("Vltava River", "the vltava river") == 1.0

    # Partial match
    # pred: ["vltava"], gold: ["vltava", "river"] -> precision 1/1, recall 1/2 -> f1 = 2/3 = 0.6667
    f1 = compute_token_f1("Vltava", "Vltava River")
    assert round(f1, 2) == 0.67

    # Zero overlap gives F1 = 0.0
    assert compute_token_f1("Danube", "Vltava") == 0.0

    # Both empty gives F1 = 1.0
    assert compute_token_f1("", "") == 1.0


def test_evaluate_answer_correctness_multiple_refs() -> None:
    prediction = "the vltava river"
    gold_refs = ["Moldau", "Vltava River", "The Vltava"]

    em, f1 = evaluate_answer_correctness(prediction, gold_refs)
    assert em is True
    assert f1 == 1.0

    # No refs
    em_none, f1_none = evaluate_answer_correctness(prediction, [])
    assert em_none is False
    assert f1_none == 0.0
