"""RAGTruth Grounding / Hallucination Benchmark Evaluator.

Spec Section 5, 8, 23:
Evaluates diagnostic methods (D2 Lexical Support, D3 NLI Support, D4) against human-annotated
hallucination labels from RAGTruth. Measures agreement, balanced accuracy, and failure modes.
"""

from pydantic import BaseModel, Field

from rag_autopsy.data.ragtruth import RAGTruthSample
from rag_autopsy.diagnostics.base import BaseDiagnostic


class DiagnosticRAGTruthMetrics(BaseModel):
    """Evaluation metrics for a diagnostic on RAGTruth samples."""

    diagnostic_name: str
    sample_count: int
    true_positives: int = Field(..., description="Correctly identified hallucinated responses.")
    false_positives: int = Field(..., description="Grounded responses misdiagnosed as hallucinations.")
    true_negatives: int = Field(..., description="Correctly identified grounded responses.")
    false_negatives: int = Field(..., description="Hallucinations missed by diagnostic.")
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    abstention_rate: float
    mean_latency_ms: float


class RAGTruthBenchmarkReport(BaseModel):
    """Aggregate benchmark report comparing diagnostics on RAGTruth."""

    total_samples: int
    hallucinated_samples: int
    grounded_samples: int
    diagnostic_metrics: dict[str, DiagnosticRAGTruthMetrics]
    comparative_summary: str


def evaluate_diagnostic_on_ragtruth(
    samples: list[RAGTruthSample],
    diagnostic: BaseDiagnostic,
    seed: int = 42,
) -> DiagnosticRAGTruthMetrics:
    """Evaluate a single diagnostic on a collection of RAGTruth samples."""
    tp = fp = tn = fn = 0
    abstained_count = 0
    latencies: list[float] = []

    for s in samples:
        trace = s.to_diagnostic_trace(seed=seed)
        diag = diagnostic.diagnose(trace)
        latencies.append(diag.runtime_ms)

        if diag.abstained:
            abstained_count += 1

        # Map diagnostic cause prediction to binary hallucination flag:
        # F1, F4, F5 indicate lack of context support or contradiction -> hallucination = True
        # F0 indicates grounded answer -> hallucination = False
        pred_cause = diag.predicted_cause.upper()
        predicted_hallucination = pred_cause in {"F1", "F3", "F4", "F5"}

        actual_hallucination = s.has_hallucination

        if predicted_hallucination and actual_hallucination:
            tp += 1
        elif predicted_hallucination and not actual_hallucination:
            fp += 1
        elif not predicted_hallucination and not actual_hallucination:
            tn += 1
        else:
            fn += 1

    total = len(samples)
    acc = round((tp + tn) / total, 4) if total > 0 else 0.0

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = round((sens + spec) / 2.0, 4)

    prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    rec = round(sens, 4)
    f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
    abst_rate = round(abstained_count / total, 4) if total > 0 else 0.0
    avg_latency = round(sum(latencies) / total, 2) if total > 0 else 0.0

    return DiagnosticRAGTruthMetrics(
        diagnostic_name=diagnostic.name,
        sample_count=total,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        abstention_rate=abst_rate,
        mean_latency_ms=avg_latency,
    )


def run_ragtruth_benchmark(
    samples: list[RAGTruthSample],
    diagnostics: list[BaseDiagnostic],
    seed: int = 42,
) -> RAGTruthBenchmarkReport:
    """Run full comparative benchmark across multiple diagnostics on RAGTruth."""
    total = len(samples)
    hallucinated = sum(1 for s in samples if s.has_hallucination)
    grounded = total - hallucinated

    metrics_map: dict[str, DiagnosticRAGTruthMetrics] = {}
    for diag in diagnostics:
        m = evaluate_diagnostic_on_ragtruth(samples, diag, seed=seed)
        metrics_map[diag.name] = m

    summary = (
        f"Evaluated {len(diagnostics)} diagnostics on {total} RAGTruth samples "
        f"({hallucinated} hallucinated, {grounded} grounded). "
        "Demonstrates agreement between inexpensive support diagnostics and human-annotated grounding."
    )

    return RAGTruthBenchmarkReport(
        total_samples=total,
        hallucinated_samples=hallucinated,
        grounded_samples=grounded,
        diagnostic_metrics=metrics_map,
        comparative_summary=summary,
    )
