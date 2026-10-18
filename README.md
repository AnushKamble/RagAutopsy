# RAG-Autopsy

**Fault-Injection-Validated Failure Attribution for RAG Systems**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Can low-cost, visible-trace diagnostics reliably attribute observed RAG failures to specific pipeline interventions without access to privileged evaluator metadata, and do those attribution patterns transfer to a sample of unmodified (natural) failures?

---

## What RAG-Autopsy Is (and Is Not)

- **IS:** A rigorous, reproducible experimental benchmark that injects controlled failure modes into a compact RAG QA pipeline, assesses whether cheap diagnostics can recover the intervention label under strict leakage boundaries, and audits transfer to natural failures.
- **IS NOT:** A commercial RAG chatbot, a generic metrics dashboard, an autonomous repair agent, or a wrapper around existing evaluation suites.

---

## Empirical Benchmark Results

Evaluated on the **held-out test split** ($N=20$ source questions $\times$ 5 controlled interventions = 100 trials) using $B=1,000$ question-level clustered bootstrap iterations (95% percentile confidence intervals):

### Attribution Benchmark (Regime A: Candidate Pool Observability)

| Diagnostic Method | Macro-F1 (95% CI) | Balanced Acc (95% CI) | Coverage | Accuracy (Covered) |
|---|---|---|---|---|
| **Majority-Cause Baseline** | 0.069 [0.067, 0.073] | 0.200 [0.200, 0.200] | 100.0% | 20.8% |
| **Random Baseline** | 0.194 [0.129, 0.255] | 0.196 [0.130, 0.265] | 100.0% | 19.8% |
| **D1 Retrieval Diagnostic** | 0.477 [0.444, 0.503] | 0.580 [0.550, 0.600] | 100.0% | 58.3% |
| **D2 Lexical Support Diagnostic** | 0.468 [0.420, 0.500] | 0.569 [0.517, 0.600] | 100.0% | 55.2% |
| **D4 Combined Policy** | **0.701 [0.651, 0.733]** | **0.769 [0.717, 0.800]** | 76.0% | **76.0%** |

### Attribution Benchmark (Regime B: Blind Top-$k$ Observability)

| Diagnostic Method | Macro-F1 (95% CI) | Balanced Acc (95% CI) | Coverage | Accuracy (Covered) |
|---|---|---|---|---|
| **D1 Retrieval Diagnostic** | 0.257 [0.224, 0.282] | 0.380 [0.350, 0.400] | 100.0% | 37.5% |
| **D4 Combined Policy** | 0.468 [0.420, 0.500] | 0.569 [0.517, 0.600] | 100.0% | 55.2% |

*Key Empirical Finding:* Without candidate pool observability (Regime B), low-cost diagnostics cannot separate evidence unavailability (F1) from rank shortfall (F2), causing Macro-F1 to drop by 23.3 percentage points. In Regime B, D4 abstains on retrieval ambiguity, preserving scientific validity over false precision.

### Extended 6-Class Benchmark (F0–F5 with NLI & Learned Policies)

Evaluated on the held-out test split ($N=120$ trials, $B=100$ bootstraps) with the complete 6-class intervention taxonomy including `F5` Unsupported Generation, `D3` NLI Support, and `D4` Learned Decision Tree:

| Diagnostic Method | Macro-F1 (95% CI) | Balanced Acc (95% CI) | Coverage | Accuracy (Covered) |
|---|---|---|---|---|
| **Majority Baseline** | 0.050 [0.048, 0.053] | 0.167 [0.167, 0.167] | 100.0% | 17.5% |
| **Random Baseline** | 0.169 [0.099, 0.264] | 0.166 [0.100, 0.265] | 100.0% | 16.7% |
| **D1 Retrieval Diagnostic** | 0.379 [0.357, 0.403] | 0.483 [0.467, 0.500] | 100.0% | 49.1% |
| **D2 Lexical Support Diagnostic** | 0.385 [0.349, 0.416] | 0.474 [0.433, 0.500] | 100.0% | 46.5% |
| **D3 NLI Support Diagnostic** | 0.374 [0.328, 0.427] | 0.447 [0.400, 0.483] | 100.0% | 43.9% |
| **D4 Combined Policy (Heuristic)** | 0.718 [0.663, 0.761] | 0.761 [0.703, 0.807] | 97.4% | 77.5% |
| **D4 Learned Policy (Decision Tree)** | **0.765 [0.745, 0.778]** | **0.825 [0.808, 0.833]** | **100.0%** | **81.6%** |

*Takeaway:* The trained Decision Tree policy outperforms the rule-based heuristic precedence by 4.7 Macro-F1 points by learning optimal joint score-cliff and lexical-overlap boundaries directly from unprivileged traces, without any data leakage.

### Natural Failure Audit ($N=100$ Unmodified Clean Questions)
- **Total Clean Questions Evaluated:** 100
- **Total Natural Failures Identified:** 100
- **Adjudicated Primary Failure Causes:**
  - `REASONING_ERROR`: 90 (90%) — Evidence reached context, but the extractor/generator failed multi-hop synthesis.
  - `F2 (Ranked Out)`: 9 (9%) — Evidence in corpus, but ranked below top-$k$.
  - `F1 (Unavailable)`: 1 (1%) — Evidence absent from corpus pool.
- **Diagnostic Transfer Agreement Rate:** 0.0% (Highlighting that pipeline transit diagnostics do not transfer to reasoning-dominated generator failures without deeper semantic judges).

---

## Architecture & Data Isolation

RAG-Autopsy strictly decouples runtime operational observability from privileged evaluation keys:
- **`DiagnosticTrace` (Diagnostic-Visible):** Top-$k$ retrieved passages, ranks, scores, candidate pool summary statistics, pre/post context assembly passage IDs, generator context, generated answer, and latency.
- **`EvaluatorRecord` (Privileged / Evaluator-Only):** Injected fault label (`F0`–`F5`), injection parameters, gold answer strings, gold evidence identifiers, and adjudication flags.

```text
Question + Corpus ---> Retriever ---> Context Assembler ---> Generator
                            |                  |                 |
                            +------------------+-----------------+
                                               |
                                     [DiagnosticTrace]
                                               |
                                               v
                                      [Diagnostic Pipeline]
                                 (D1: Retrieval, D2: Lexical, D3: NLI)
                                               |
                                               v
                                          [Diagnosis]
                                               |
               [EvaluatorRecord]               v
                       |             [Attribution Evaluation]
                       +-----------------------> Macro-F1 / Confusion Matrix / Bootstrap CIs
```

---

## Quickstart & Reproduction

### Requirements
- Python 3.11
- [`uv`](https://github.com/astral-sh/uv)

### Setup Virtual Environment
```bash
# Create virtual environment
uv venv --python 3.11

# Install package in editable mode with development dependencies
uv pip install -e ".[dev]"
```

### Reproduce Benchmarks, Dashboards & Visualizations
```bash
# 1. Run all 110 automated unit, integration, and leakage tests (~25 seconds)
pytest -v

# 2. Run the 60-second interactive live walkthrough demo
python demo.py
# (or via CLI)
rag-autopsy demo

# 3. Launch the Interactive Streamlit Studio (Real-time injection and inspection)
streamlit run app.py

# 4. Generate standalone zero-dependency interactive HTML evaluation report
rag-autopsy report --output results/report.html

# 5. Generate static publication-quality confusion matrices and bootstrap CI plots
rag-autopsy generate-plots

# 6. Run full 6-class benchmark with F5, D3 NLI, and D4 Learned Classifier
rag-autopsy evaluate --data .data/processed/hotpot_subset_100.json --split test --regime regime_a --bootstraps 100 --include-f5 --include-d3 --include-d4-learned

# 7. Run natural failure audit
rag-autopsy audit-natural --data .data/processed/hotpot_subset_100.json

# 8. Run 5-axis systematic ablation study suite (Spec Section 12)
rag-autopsy ablate

# 9. Run fault severity sweeps (distractor counts & context budget)
rag-autopsy severity

# 10. Run RAGTruth grounding & hallucination benchmark (D2 vs D3 on human annotations)
rag-autopsy ragtruth

# 11. Run RAG quality baseline comparison (Ragas Context Precision & Faithfulness vs Attribution)
rag-autopsy quality-compare
```

### Generated Visualizations & Reports
- `results/report.html`: Standalone interactive evaluation dashboard with metric cards and trial explorer.
- `results/plots/regime_a_d4_confusion_matrix.png`: Normalized attribution confusion matrix for D4 Combined Policy in Regime A.
- `results/plots/regime_comparison_d4.png`: Side-by-side Regime A vs Regime B comparison illustrating identifiability collapse on F2.
- `results/plots/regime_a_macro_f1_bootstrap.png`: Macro-F1 benchmark comparison with 95% clustered bootstrap error bars.
- `results/ablations/ablation_study_report.json`: Multi-axis ablation results across diagnostics, abstention, compound faults, and severity.
- `results/severity/severity_report.json`: Sensitivity curves across distractor density ($D \in [1..20]$) and context length limits ($C \in [500..3000]$).
- `results/ragtruth/ragtruth_benchmark_report.json`: Support diagnostic benchmarking against human hallucination annotations.
- `results/rag_quality/quality_vs_attribution.json`: Scalar quality metric profile contrasting precision/faithfulness against stage attribution.

---

## Documentation
- [`docs/project_charter.md`](docs/project_charter.md) — Research motivation, questions, and positioning.
- [`docs/data_contract.md`](docs/data_contract.md) — HotpotQA terms, schemas, and partitioning invariants.
- [`docs/trace_schema.md`](docs/trace_schema.md) — Data contracts, trace schemas, and boundary isolation audit.
- [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) — Frozen pre-registered evaluation protocol.
- [`docs/annotation_guide.md`](docs/annotation_guide.md) — Human reviewer rubric for auditing natural failures.
- [`docs/decisions.md`](docs/decisions.md) — Architecture Decision Records (ADRs).
- [`docs/limitations.md`](docs/limitations.md) — Detailed threats to validity and scope bounds.
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — Live milestone verification tracking.


