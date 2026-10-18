"""Unit tests for RAGTruth adapter and evaluation harness."""

from pathlib import Path
from rag_autopsy.data.ragtruth import RAGTruthSample, load_ragtruth_json
from rag_autopsy.diagnostics.d2_support import D2LexicalSupportDiagnostic
from rag_autopsy.diagnostics.d3_nli import D3NLISupportDiagnostic
from rag_autopsy.evaluation.ragtruth_eval import (
    evaluate_diagnostic_on_ragtruth,
    run_ragtruth_benchmark,
)


def test_ragtruth_loader_and_trace_conversion(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/ragtruth_synthetic.json")
    samples = load_ragtruth_json(fixture_path)

    assert len(samples) == 5
    s0 = samples[0]
    assert s0.sample_id == "rt_syn_001"
    assert s0.has_hallucination is False

    # Test conversion to DiagnosticTrace
    trace = s0.to_diagnostic_trace(seed=42)
    assert trace.question_id == "rt_syn_001"
    assert "Apollo 11" in trace.generator_context
    assert trace.answer == s0.response
    assert len(trace.retrieved_top_k) == 1

    # Ensure no leakage keys in trace
    assert not hasattr(trace, "has_hallucination")
    assert not hasattr(trace, "hallucination_spans")


def test_ragtruth_benchmark_evaluation() -> None:
    fixture_path = Path("tests/fixtures/ragtruth_synthetic.json")
    samples = load_ragtruth_json(fixture_path)

    d2 = D2LexicalSupportDiagnostic()
    d3 = D3NLISupportDiagnostic()

    report = run_ragtruth_benchmark(samples, [d2, d3], seed=42)

    assert report.total_samples == 5
    assert report.hallucinated_samples == 3
    assert report.grounded_samples == 2
    assert d2.name in report.diagnostic_metrics
    assert d3.name in report.diagnostic_metrics

    # D2 metrics (lexical overlap misses subtle entity/number substitutions)
    m2 = report.diagnostic_metrics[d2.name]
    assert m2.sample_count == 5
    assert m2.accuracy == 0.4  # Correctly identifies 2 grounded, misses 3 fact substitutions

    # D3 metrics (NLI support identifies factual divergence)
    m3 = report.diagnostic_metrics[d3.name]
    assert m3.sample_count == 5
    assert m3.accuracy >= 0.8  # Strong hallucination detection
    assert m3.recall == 1.0    # Recalls all human-annotated hallucinations
    assert m3.f1_score >= 0.80
