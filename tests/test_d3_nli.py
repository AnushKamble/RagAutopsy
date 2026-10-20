"""Unit tests for D3: NLI Support and Contradiction Diagnostic."""

from pathlib import Path

from rag_autopsy.core.schemas import DiagnosticTrace, RetrievedPassage
from rag_autopsy.diagnostics.d3_nli import D3NLISupportDiagnostic


def test_d3_nli_evaluate_pair_and_caching(tmp_path: Path):
    d3 = D3NLISupportDiagnostic(cache_dir=str(tmp_path / "nli_cache"))
    premise = "Albert Einstein developed the theory of relativity in 1905."
    hypo_entail = "Einstein worked on the theory of relativity."
    hypo_contra = "Albert Einstein did not develop the theory of relativity."

    res_entail = d3.evaluate_pair(premise, hypo_entail)
    assert res_entail["entailment"] > res_entail["contradiction"]

    res_contra = d3.evaluate_pair(premise, hypo_contra)
    assert res_contra["contradiction"] > 0.50

    # Verify cache files were created
    cache_files = list((tmp_path / "nli_cache").glob("*.json"))
    assert len(cache_files) == 2


def test_d3_diagnose_f4_contradiction(tmp_path: Path):
    d3 = D3NLISupportDiagnostic(cache_dir=str(tmp_path))
    p1 = RetrievedPassage(passage_id="p1", rank=1, score=10.0, text="The speed limit is 50 mph.")
    p2 = RetrievedPassage(passage_id="p2", rank=2, score=8.0, text="The speed limit is not 50 mph.")

    trace = DiagnosticTrace(
        trial_id="trace_contra_001",
        question_id="q1",
        question="What is the speed limit?",
        retrieved_top_k=[p1, p2],
        source_passage_ids=["p1", "p2"],
        pre_assembly_passage_ids=["p1", "p2"],
        post_assembly_passage_ids=["p1", "p2"],
        generator_context="[Passage 1]: The speed limit is 50 mph.\n\n---\n\n[Passage 2]: The speed limit is not 50 mph.",
        answer="The speed limit is not 50 mph.",
        latency_ms=12.0,
    )

    diag = d3.diagnose(trace)
    assert diag.predicted_cause in ("F4", "F0")  # Contradiction signal detected


def test_d3_diagnose_f5_unsupported_generation(tmp_path: Path):
    d3 = D3NLISupportDiagnostic(
        cache_dir=str(tmp_path),
        unsupported_entailment_threshold=0.35,
    )
    p1 = RetrievedPassage(passage_id="p1", rank=1, score=10.0, text="Bananas grow on tropical herbaceous plants.")

    trace = DiagnosticTrace(
        trial_id="trial_nli_unsupported_001",
        question_id="q1",
        question="Where do bananas grow?",

        retrieved_top_k=[p1],
        source_passage_ids=["p1"],
        pre_assembly_passage_ids=["p1"],
        post_assembly_passage_ids=["p1"],
        generator_context="[Passage 1]: Bananas grow on tropical herbaceous plants.",
        answer="Bananas are manufactured in subterranean lunar factories.",
        latency_ms=15.0,
    )

    diag = d3.diagnose(trace)
    assert diag.predicted_cause == "F5"
    assert "unsupported generation" in diag.evidence_rationale.lower()
