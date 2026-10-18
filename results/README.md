# Results Directory & Experiment Artifacts

This directory stores all reproducible benchmark outputs, evaluation summaries, uncertainty intervals, figures, and interactive reports for RAG-Autopsy.

---

## 1. Directory Structure

```text
results/
  report.html                              # Standalone interactive zero-dependency evaluation dashboard
  test_regime_a/                           # Held-out test evaluation under Regime A (Candidate Pool Trace)
    evaluation_summary.json                # Summary metrics with 95% clustered bootstrap intervals
    trial_predictions.csv                  # Per-trial predicted vs true causes and rationales
  test_regime_b/                           # Held-out test evaluation under Regime B (Blind Top-k)
    evaluation_summary.json
    trial_predictions.csv
  test_regime_a_extended/                  # 6-class benchmark (F0–F5) with D3 NLI and D4 Learned Classifier
    evaluation_summary.json
    trial_predictions.csv
  natural_audit/                           # Unmodified clean pipeline natural failure audit
    natural_audit_summary.json             # Root-cause distributions and transfer agreement results
  ablations/                               # 5-axis systematic ablation study suite outputs
    ablation_study_report.json             # Predeclared ablation metrics (diagnostics, abstention, etc.)
  severity/                                # Fault severity scaling & sensitivity analysis
    severity_report.json                   # Distractor count & context budget sensitivity curves
  ragtruth/                                # RAGTruth grounding & hallucination benchmark results
    ragtruth_benchmark_report.json         # D2 vs D3 agreement with human hallucination annotations
  rag_quality/                             # Conventional RAG quality comparison (Ragas)
    quality_vs_attribution.json            # Context precision & faithfulness breakdown vs attribution
  plots/                                   # Static publication figures
    regime_a_d4_confusion_matrix.png       # Normalized confusion matrix heatmap
    regime_comparison_d4.png               # Regime A vs Regime B identifiability comparison
    regime_a_macro_f1_bootstrap.png        # Macro-F1 with 95% clustered bootstrap error bars
  pilot/                                   # Dedicated injector validation pilot reports
    pilot_f1_report.json                   # Pre/post trace inspection for F1 Evidence Unavailable
```

---

## 2. Artifact Schemas

### `evaluation_summary.json`
- `target_split`: Name of partition (`dev`, `calibration`, `test`).
- `regime`: Observability regime (`regime_a` or `regime_b`).
- `validity_summary`: Counts of attempted, valid, and invalid trials with adjudication reasons.
- `diagnostic_results`: Per-diagnostic attribution metrics (`macro_f1`, `balanced_accuracy`, `confusion_matrix`, `coverage`, `per_class_metrics`) and question-level clustered bootstrap percentile intervals.

### `trial_predictions.csv`
- `trial_id`: 16-hex opaque identifier (e.g. `trial_37340db14bdc8d71`).
- `question_id`: Stable question identifier.
- `true_cause`: Evaluator ground-truth intervention label (`F0`–`F5`).
- `predicted_cause`: Diagnoser prediction.
- `abstained`: Boolean indicating diagnostic abstention.
- `confidence`: Calibrated diagnostic confidence $[0.0, 1.0]$.
- `rationale`: Human-interpretable trace rationale.
