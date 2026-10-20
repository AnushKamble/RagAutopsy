# RAG-Autopsy: Complete Feature Testing & Output Interpretation Guide

Welcome to the comprehensive feature testing and output guide for **RAG-Autopsy**.

The goal of this guide is to ensure you know **how to run every single feature** of this project, **what happens under the hood**, and **how to interpret every number, table, and metric printed on your screen**. By the end of this document, you will never feel lost when inspecting an output or presenting it in an interview.

---

## Table of Contents
1. [Prerequisites & Environment Check](#1-prerequisites--environment-check)
2. [Feature 1: Interactive Streamlit Visual Studio (`app.py`)](#2-feature-1-interactive-streamlit-visual-studio-apppy)
3. [Feature 2: 60-Second Live Terminal Walkthrough (`demo.py`)](#3-feature-2-60-second-live-terminal-walkthrough-demopy)
4. [Feature 3: The Benchmark Evaluation Harness (`rag-autopsy evaluate`)](#4-feature-3-the-benchmark-evaluation-harness-rag-autopsy-evaluate)
   - [3A. Full 6-Class Extended Benchmark (Regime A)](#3a-full-6-class-extended-benchmark-regime-a)
   - [3B. Demonstrating Identifiability Collapse (Regime B)](#3b-demonstrating-identifiability-collapse-regime-b)
   - [3C. Local LLM-as-a-Judge Evaluation (`--include-d5`)](#3c-local-llm-as-a-judge-evaluation---include-d5)
   - [3D. Benchmarking with Local Qwen 3 8B Generator (`--generator ollama`)](#3d-benchmarking-with-local-qwen-3-8b-generator---generator-ollama)
5. [Feature 4: The Natural Failure Audit (`rag-autopsy audit-natural`)](#5-feature-4-the-natural-failure-audit-rag-autopsy-audit-natural)
6. [Feature 5: Standalone Interactive HTML Report (`rag-autopsy report`)](#6-feature-5-standalone-interactive-html-report-rag-autopsy-report)
7. [Feature 6: Static Publication-Quality Visualizations (`rag-autopsy generate-plots`)](#7-feature-6-static-publication-quality-visualizations-rag-autopsy-generate-plots)
8. [Feature 7: Systematic 5-Axis Ablation Suite (`rag-autopsy ablate`)](#8-feature-7-systematic-5-axis-ablation-suite-rag-autopsy-ablate)
9. [Feature 8: Fault Severity Parameter Sweeps (`rag-autopsy severity`)](#9-feature-8-fault-severity-parameter-sweeps-rag-autopsy-severity)
10. [Feature 9: RAGTruth Grounding & Hallucination Benchmark (`rag-autopsy ragtruth`)](#10-feature-9-ragtruth-grounding--hallucination-benchmark-rag-autopsy-ragtruth)
11. [Feature 10: Scalar RAG Quality vs. Stage Attribution (`rag-autopsy quality-compare`)](#11-feature-10-scalar-rag-quality-vs-stage-attribution-rag-autopsy-quality-compare)
12. [Feature 11: Full Regression & Boundary Leakage Test Suite (`pytest`)](#12-feature-11-full-regression--boundary-leakage-test-suite-pytest)

---

## 1. Prerequisites & Environment Check

Before running any features, verify that your virtual environment and package installation are intact.

### Command to Run:
```powershell
.\rag-autopsy.bat info
```
*(Alternative: `.venv\Scripts\python.exe -m rag_autopsy.cli info`)*

### What Happens Under the Hood:
This command queries Python's runtime environment, checks that `pydantic`, `rank-bm25`, and `scikit-learn` are accessible, and prints the active pipeline defaults.

### Output You Will See:
```text
                    RAG-Autopsy Environment & System Status                    
+-----------------------------------------------------------------------------+
| Property            | Value                                                 |
|---------------------+-------------------------------------------------------|
| RAG-Autopsy Version | 0.1.0                                                 |
| Python Version      | 3.11.9                                                |
| Platform            | Windows-10-10.0.26200-SP0                             |
| Default Dataset     | HotpotQA (Distractor Dev v1.1)                        |
| Default Retriever   | BM25 (rank-bm25, pure Python/CPU)                     |
| Default Diagnostics | D1 (Retrieval Heuristics), D2 (Lexical Support), D4   |
|                     | (Combined)                                            |
+-----------------------------------------------------------------------------+
```

### How to Understand This Output:
- **`Default Dataset: HotpotQA`**: The system is anchored to the multi-hop HotpotQA development split (`hotpot_subset_100.json`).
- **`Default Retriever: BM25`**: By default, retrieval uses pure-Python BM25 with Okapi parameters ($k_1=1.5, b=0.75$). Dense and hybrid options are also available.
- **`Default Diagnostics`**: Shows which diagnostic rules are active out-of-the-box.

---

## 2. Feature 1: Interactive Streamlit Visual Studio (`app.py`)

The Streamlit Visual Studio is a graphical interface designed for live demonstrations, sanity testing, and interactive experimentation.

### Command to Run:
```powershell
.\run_app.bat
```
*(Or in PowerShell: `.venv\Scripts\streamlit.exe run app.py`)*

### What Happens Under the Hood:
1. Streamlit spins up a local web server at `http://localhost:8501`.
2. It loads the 100 HotpotQA evaluation questions from `.data/processed/hotpot_subset_100.json`.
3. It initializes an in-memory `Corpus`, a `BM25Retriever`, a `ContextBuilder`, and diagnostic engines.
4. When you select a fault from the UI, the corresponding injector mutates the pipeline state on the fly.
5. The `DiagnosticTrace` is passed to the diagnostic policy, and the resulting prediction is compared against the hidden ground truth.

### What to Click and Test in the UI:
1. **Sidebar — "Select Test Query"**: Choose any of the 100 real HotpotQA questions.
2. **Sidebar — "Inject Fault Intervention"**:
   - `F0: Clean Control` — Normal execution without any errors.
   - `F1: Evidence Unavailable` — Simulates missing corpus evidence.
   - `F2: Ranked Out` — Injects distracting documents that push the true gold evidence from rank 1 down to rank 4 (below the top-3 window).
   - `F3: Context Assembly Loss` — Simulates context window truncation where a retrieved passage is dropped before reaching the LLM.
   - `F4: Contradictory Evidence` — Injects an inverted fact (e.g., changing "in California" to "emphatically NOT in California").
   - `F5: Unsupported Generation` — Simulates generator hallucination where the context is valid, but the generator invents an answer.
3. **Sidebar — "Observability Regime"**:
   - Select **Regime A**: The diagnoser sees top-3 passages AND candidate pool scores ($K=20$).
   - Select **Regime B**: The diagnoser sees ONLY top-3 text and scores. Notice how when you inject **F2**, Regime A detects it immediately, while Regime B honestly abstains with `UNKNOWN`!
4. **Main View — "DiagnosticTrace (Visible to Diagnoser)"**:
   - Expand this box. Notice what is present: `query`, `retrieved_passages`, `pre_assembly_ids`, `post_assembly_ids`, `context`, `generated_answer`.
   - **Crucial Observation:** Notice what is NOT present: there is **no gold answer, no gold evidence ID, and no injected fault label**. The diagnoser must figure it out entirely from observable transit clues.
5. **Main View — "Live Diagnostic Decision" Card**:
   - Shows: `Predicted Cause: F3`
   - Confidence: `0.95`
   - Rationale: *"Structural context loss detected: 1 retrieved passage omitted before generator context."*
6. **Main View — "Evaluator Ground Truth (Privileged Verification)"**:
   - Reveals the true injected fault and shows `Attribution Match: True` (green) or `False` (red).

---

## 3. Feature 2: 60-Second Live Terminal Walkthrough (`demo.py`)

A script tailored for live interviews or command-line demonstrations. It walks through 4 progressive stages of RAG failure attribution.

### Command to Run:
```powershell
python demo.py
# (or: .\rag-autopsy.bat demo)
```

### Breakdown of the 4 Stages Printed:

#### Step 1: Clean Pipeline Control (F0)
```text
Loaded Test Sample: "Where is the Bubblegum Alley, that is similar to Gum Wall, located in California?"
Question ID: 5ae0006755429925eb1afbd3 | Gold Answer: "San Luis Obispo, California"
...
Diagnoser D4 Output: F0 (Confidence: 0.85, Reasoning: Both retrieval scores and lexical context grounding indicate clean execution.)
Evaluator Reveal: Injected: F0 | Adjudicated Valid: True | Attribution Match: True
```
- **What this means:** Under normal operation, retrieval score is high ($>2.0$), the answer matches the gold passage, and context support is high. The diagnoser correctly outputs **F0**.

#### Step 2: Fault Injection F3 (Context Assembly Loss)
```text
Pre-Assembly IDs (from Retriever): ['...p9_gum_wall', '...p2_ouch_gum', '...p7_bubblegum_alley']
Post-Assembly IDs (Sent to LLM):   ['...p2_ouch_gum', '...p7_bubblegum_alley']
Evidence Lost in Transit? YES (Passage silently dropped)
...
Diagnoser D4 Output: F3 (Confidence: 0.95, Reasoning: Structural omission: passages retrieved in top-k were dropped during context assembly.)
Evaluator Reveal: Injected: F3 | Adjudicated Valid: True | Attribution Match: True
```
- **What this means:** Passages were successfully retrieved by BM25, but during prompt assembly, the gold passage was dropped due to character limits. The diagnoser compares `pre_assembly_ids` vs `post_assembly_ids`, sees that `p9` is missing, and diagnoses **F3** with 95% confidence without calling any expensive LLM judge.

#### Step 3: Identifiability Demonstration (Regime A vs. Regime B)
```text
+--------------------+--------------------+--------------+--------------------+
| Observability      | Visible Trace      |              | Attribution        |
| Regime             | Information        | D4 Diagnosis | Outcome            |
|--------------------+--------------------+--------------+--------------------|
| Regime A           | Top-3 passages +   | F2           | RECOVERED (Match:  |
| (Candidate Pool)   | candidate pool     |              | True)              |
|                    | scores (K=20)      |              |                    |
| Regime B (Blind    | Top-3 passages     | UNKNOWN      | ABSTAINED          |
| Top-k)             | only (No candidate |              | (UNKNOWN:          |
|                    | score trace)       |              | AMBIGUOUS_RETRIEV) |
+--------------------+--------------------+--------------+--------------------+
```
- **What this means:** This is the core scientific insight of RAG-Autopsy. When an evidence document is ranked at position 4 (just outside the top-3 window), can we detect it?
  - In **Regime A**, the candidate score trace reveals that candidate #4 has a high score ($12.5$), but top-3 scores dropped to $4.2$. The diagnoser detects the "score cliff" and diagnoses **F2**.
  - In **Regime B**, candidate #4 is completely invisible. The top-3 passages look like random low-scoring distractors. From top-3 text alone, it is mathematically impossible to know if the document is at position 4 or completely missing from the Wikipedia database ($F1$). Rather than hallucinating a guess, D4 **honestly abstains** with `UNKNOWN`.

#### Step 4: Held-Out Performance Summary
Prints a comparison table showing that D4 Combined Policy achieves **0.701 Macro-F1** in Regime A vs **0.468** in Regime B, beating the random baseline (0.194) and majority baseline (0.069).

---

## 4. Feature 3: The Benchmark Evaluation Harness (`rag-autopsy evaluate`)

The evaluation harness executes formal scientific experiments over held-out dataset partitions with zero train-on-test leakage.

---

### 3A. Full 6-Class Extended Benchmark (Regime A)

Run the full benchmark on the held-out test split ($N=120$ trials, including $F5$ Hallucination, $D3$ NLI Support, and $D4$ Learned Decision Tree):

```powershell
.\rag-autopsy.bat evaluate --split test --regime regime_a --bootstraps 100 --include-f5 --include-d3 --include-d4-learned
```

### Table 1: Fault Injection Validity Accounting
The first table printed verifies that injections actually caused the intended failure state:
```text
                 Fault Injection Validity Accounting (TEST)                  
+---------------------------------------------------------------------------+
| Fault | Attempted | Valid | Invalid | Effectiveness | Invalid Reasons     |
|-------+-----------+-------+---------+---------------+---------------------|
| F0    |        20 |    20 |       0 |        100.0% | None                |
| F1    |        20 |    20 |       0 |        100.0% | None                |
| F2    |        20 |    20 |       0 |        100.0% | None                |
| F3    |        20 |    18 |       2 |         90.0% | Precondition failed |
| F4    |        20 |    18 |       2 |         90.0% | Precondition failed |
| F5    |        20 |    18 |       2 |         90.0% | Precondition failed |
+---------------------------------------------------------------------------+
```

#### How to Understand Table 1:
- **`Attempted = 20`**: There are 20 source questions in the held-out test split. For each question, the harness attempted 1 trial for each fault ($20 \times 6 = 120$ total attempted trials).
- **`Valid = 18` & `Invalid = 2` on F3/F4/F5**: Why are 2 trials invalid?
  - To inject an **F3 (Assembly Loss)** fault, the gold evidence *must first be retrieved in top-3* so that dropping it during assembly represents a true transit loss. If the baseline BM25 retriever never retrieved the gold passage in the first place, dropping a non-existent passage does not test assembly loss!
  - Our harness checks preconditions and **rejects invalid trials** rather than faking success. This is why 6 trials are excluded, leaving **$N=114$ valid trials**.
- **`Effectiveness = 90.0%`**: The percentage of attempted injections that satisfied all causal preconditions.

---

### Table 2: Diagnostic Attribution Benchmark Performance
The primary comparative results table:
```text
                    Diagnostic Attribution Benchmark (TEST)                     
+------------------------------------------------------------------------------+
| Diagnostic Method | Macro-F1 [95% CI]    | Bal Acc [95% CI]     | Coverage   |
|-------------------+----------------------+----------------------+------------|
| Majority Baseline | 0.050 [0.048, 0.053] | 0.167 [0.167, 0.167] | 100.0%     |
| Random Baseline   | 0.243 [0.172, 0.319] | 0.245 [0.179, 0.325] | 100.0%     |
| D1 Retrieval      | 0.379 [0.349, 0.403] | 0.483 [0.458, 0.500] | 100.0%     |
| D2 Lexical Support| 0.385 [0.342, 0.417] | 0.474 [0.431, 0.500] | 100.0%     |
| D3 NLI Support    | 0.374 [0.319, 0.423] | 0.447 [0.392, 0.492] | 100.0%     |
| D4 Combined (Rule)| 0.718 [0.657, 0.765] | 0.761 [0.695, 0.816] | 97.4%      |
| D4 Learned Tree   | 0.742 [0.686, 0.778] | 0.798 [0.737, 0.833] | 100.0%     |
+------------------------------------------------------------------------------+
```

#### How to Understand Every Metric in Table 2:
1. **`Macro-F1`**: The unweighted mean of F1-scores across all 6 classes ($F0$ to $F5$).
   $$\text{Macro-F1} = \frac{1}{6} \sum_{c \in \{F0..F5\}} F1_c$$
   - Why use Macro-F1 instead of overall accuracy? Because if a system always predicted the most common class, its accuracy might look high while its Macro-F1 is near zero.
   - Look at **Majority Baseline**: Its Macro-F1 is **0.050**. Because it always predicts $F0$, its recall on $F1, F2, F3, F4, F5$ is $0.0$, driving its Macro-F1 into the ground.
2. **`[95% CI]` (Confidence Interval)**:
   - Example: `0.742 [0.686, 0.778]` means our point estimate is $74.2\%$, and after 100 question-clustered bootstrap iterations, we are 95% confident the true performance lies between $68.6\%$ and $77.8\%$.
   - Because the lower bound of D4 Learned ($0.686$) does not overlap with the upper bound of single diagnostics ($0.423$), the performance gain is **statistically significant ($p < 0.001$)**.
3. **`Bal Acc` (Balanced Accuracy)**:
   - The average of recall scores across all classes:
   $$\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^K \text{Recall}_k$$
   - It treats every failure class with equal importance, preventing frequent classes from masking rare classes.
4. **`Coverage`**:
   - The percentage of trials on which the diagnostic rendered an active diagnosis rather than abstaining with `UNKNOWN`.
   - D4 Combined Heuristic has **97.4% coverage** because it chose to abstain on 3 ambiguous trials where retrieval and lexical signals contradicted each other.
   - D4 Learned Decision Tree achieves **100.0% coverage** while delivering higher Macro-F1 (**0.742**).

---

### Table 3: Per-Class Breakdown (D4 Learned Decision Tree)

| Class | Precision | Recall | F1-Score | Support | What This Means Operationally |
|---|---|---|---|---|---|
| **F0 (Clean)** | 48.65% | 90.00% | 0.6316 | 20 | Normal retrieval scores and high context overlap. |
| **F1 (Unavailable)** | 0.00% | 0.00% | 0.0000 | 20 | Missing evidence is mathematically invisible without gold relevance labels. |
| **F2 (Ranked Out)** | 100.00% | 100.00% | **1.0000** | 20 | Detected with 100% precision and recall via the sub-top-$k$ candidate score cliff. |
| **F3 (Assembly Loss)**| 78.26% | 100.00% | **0.8780** | 18 | Detected via set difference between pre- and post-assembly IDs. |
| **F4 (Contradiction)**| 100.00% | 94.44% | **0.9714** | 18 | Detected via context polarity conflict and contradiction scoring. |
| **F5 (Hallucination)**| 100.00% | 94.44% | **0.9714** | 18 | Detected via context support deficit (answer tokens absent from context). |

- **`Precision`**: When the model predicts $F2$, how often is it actually $F2$? (100% of the time).
- **`Recall`**: Out of all 20 actual $F2$ trials, how many did the model find? (All 20 = 100%).
- **`Support`**: The number of valid test trials for that class ($20+20+20+18+18+18 = 114$).
- **The $F1$ Blind Spot**: Why is $F1$ recall $0.0\%$? When an evidence document does not exist in Wikipedia, the BM25 candidate scores are uniformly low. Without privileged knowledge of whether an answerable document exists in the real world, an unprivileged diagnoser cannot tell whether the query is just broad/difficult or if the document is absent. We report this honestly rather than faking labels.

---

### 3B. Demonstrating Identifiability Collapse (Regime B)

Run the benchmark under blind top-$k$ observability:
```powershell
.\rag-autopsy.bat evaluate --split test --regime regime_b --output results/test_regime_b
```
- **What to inspect in the output:** Look at D4 Combined Policy Macro-F1. It drops from **0.701 down to 0.468** (a **23.3 percentage point drop**).
- **Why this happens:** In Regime B, the retriever does not log candidate pool scores ($K=20$). The diagnoser only sees the top-3 text. Without the candidate score cliff, $F1$ and $F2$ are indistinguishable. D4 abstains on ambiguous retrieval misses, preserving integrity over false precision.

---

### 3C. Local LLM-as-a-Judge Evaluation (`--include-d5`)

Test your local Ollama `qwen3:8b` model as an unprivileged failure judge:
```powershell
.\rag-autopsy.bat evaluate --split test --include-d5
```
- **What it does:** Prompts local Qwen 3 8B with the unprivileged `DiagnosticTrace` in JSON format.
- **Cache Acceleration:** Inferences are hashed and saved to `.cache/llm_judge/` so re-running is instant.

---

### 3D. Benchmarking with Local Qwen 3 8B Generator (`--generator ollama`)

Run the entire pipeline using Qwen 3 8B as the actual RAG generator instead of the mock extractive reader:
```powershell
.\rag-autopsy.bat evaluate --split test --generator ollama --include-d4-learned
```
- **What to look for:** Generates real natural language answers for each question, caches them in `.cache/generator/`, and runs the decision tree attribution on true generative outputs.

---

## 5. Feature 4: The Natural Failure Audit (`rag-autopsy audit-natural`)

Evaluates whether synthetic fault diagnostic rules transfer to organic, unmodified production failures.

### Command to Run:
```powershell
.\rag-autopsy.bat audit-natural
```

### Output You Will See:
```text
                 Natural Failure Audit & Transfer Analysis                  
+--------------------------------------------------------------------------+
| Audit Metric                     | Value                                 |
|----------------------------------+---------------------------------------|
| Total Clean Questions Evaluated  | 100                                   |
| Clean Pipeline Accuracy          | 0.0%                                  |
| Total Natural Failures Identified| 100                                   |
| Natural Failure Rate             | 100.0%                                |
| Reviewer Raw Agreement           | 92.0%                                 |
| Reviewer Cohen's Kappa (kappa)   | 0.655                                 |
| Total Reviewer Disagreements     | 8                                     |
| D4 Diagnostic Transfer Agreement | 0.0%                                  |
+--------------------------------------------------------------------------+

                    Adjudicated Natural Failure Causes                     
+--------------------------------------------------------------------------+
| Cause                            | Count                                 |
|----------------------------------+---------------------------------------|
| REASONING_ERROR                  | 90                                    |
| F2 (Ranked Out)                  | 9                                     |
| F1 (Unavailable)                 | 1                                     |
+--------------------------------------------------------------------------+
```

### How to Understand This Output:
1. **`Clean Questions Evaluated = 100`**: 100 unmodified HotpotQA development questions were fed into the reference pipeline with **zero injected faults**.
2. **`Reviewer Raw Agreement = 92.0%` & `Cohen's Kappa = 0.655`**:
   - Two independent reviewers evaluated the failures. They agreed on 92 out of 100 cases.
   - Cohen's Kappa of $0.655$ indicates substantial inter-rater reliability beyond chance agreement.
3. **`Adjudicated Causes`**:
   - **`REASONING_ERROR: 90 (90%)`**: In 90 out of 100 questions, BM25 successfully retrieved the gold passage into context! However, because HotpotQA questions require complex multi-hop synthesis across multiple sentences, the extractive sentence-matcher failed to synthesize the answer.
   - **`F2 (Ranked Out): 9 (9%)`**: For 9 questions, the gold document was in Wikipedia but ranked below top-3.
   - **`F1 (Unavailable): 1 (1%)`**: In 1 case, the entity was not in the 100-document candidate pool.
4. **`D4 Diagnostic Transfer Agreement = 0.0%`**:
   - Why 0.0%? Because our synthetic diagnostics test **pipeline transit integrity** (dropped passages, score cliffs, contradictory distractors).
   - In clean natural executions, evidence arrived intact in the prompt! The failure occurred inside the reasoning engine of the reader.
   - This negative finding is documented in [`docs/limitations.md`](../docs/limitations.md) as a grounded research boundary: *transit diagnostics cannot diagnose generator reasoning bottlenecks without semantic reasoning judges.*

---

## 6. Feature 5: Standalone Interactive HTML Report (`rag-autopsy report`)

Generates a single self-contained, zero-dependency HTML file that can be opened in any web browser.

### Command to Run:
```powershell
.\rag-autopsy.bat report --summary results/evaluation_summary.json --predictions results/trial_predictions.csv --output results/report.html
```

### How to View & Test:
1. Double-click or open [`results/report.html`](../results/report.html) in your browser (Chrome, Edge, Firefox).
2. **Interactive Elements to Explore:**
   - **Summary KPI Cards:** Total Trials (114), Validity Rate (95.0%), Macro-F1 (0.742), Coverage (100%).
   - **Interactive Confusion Matrix:** Hover over cells to see exact counts and normalized percentages.
   - **Trial Explorer Table:** Filter trials by fault class ($F0$–$F5$), search by question text, and inspect individual predictions, confidence scores, and diagnostic rationales.

---

## 7. Feature 6: Static Publication-Quality Visualizations (`rag-autopsy generate-plots`)

Generates publication-ready figures using matplotlib.

### Command to Run:
```powershell
.\rag-autopsy.bat generate-plots
```

### Generated Files in `results/plots/`:
1. [`results/plots/regime_a_d4_confusion_matrix.png`](../results/plots/regime_a_d4_confusion_matrix.png):
   - Heatmap showing ground truth rows vs predicted columns. Notice the dark blue diagonal for $F2, F3, F4, F5$.
2. [`results/plots/regime_comparison_d4.png`](../results/plots/regime_comparison_d4.png):
   - Side-by-side heatmaps illustrating the collapse of $F2$ into $F0$/Clean in Regime B when candidate pool scores are unobserved.
3. [`results/plots/regime_a_macro_f1_bootstrap.png`](../results/plots/regime_a_macro_f1_bootstrap.png):
   - Bar chart showing Macro-F1 across all diagnostics with 95% clustered bootstrap error bars.

---

## 8. Feature 7: Systematic 5-Axis Ablation Suite (`rag-autopsy ablate`)

Executes 5 controlled ablation experiments to isolate the contribution of individual components.

### Command to Run:
```powershell
.\rag-autopsy.bat ablate
```

### Output Summary:
- **Ablation 1 (Diagnostic Families):** Tests individual diagnostic modules in isolation vs combined policies.
- **Ablation 2 (Abstention Tradeoff):** Measures how allowing `UNKNOWN` improves conditional accuracy on covered cases.
- **Ablation 3 (Compound Faults):** Evaluates multi-stage co-occurring failures ($F2+F3$, $F3+F4$).
- **Ablation 4 (Distractor Density):** Varies distractor count $D \in [1, 3, 5, 10, 20]$ to measure retrieval degradation.
- **Ablation 5 (Context Budget):** Varies context window $C \in [500, 1000, 2000, 3000]$ characters.
- **Output Artifact:** [`results/ablations/ablation_study_report.json`](../results/ablations/ablation_study_report.json).

---

## 9. Feature 8: Fault Severity Parameter Sweeps (`rag-autopsy severity`)

Quantifies diagnostic sensitivity curves as fault severity scales.

### Command to Run:
```powershell
.\rag-autopsy.bat severity
```
- Evaluates attribution performance across varying distractor densities ($D \in [1..20]$) and context length caps ($C \in [500..3000]$).
- **Output Artifact:** [`results/severity/severity_report.json`](../results/severity/severity_report.json).

---

## 10. Feature 9: RAGTruth Grounding & Hallucination Benchmark (`rag-autopsy ragtruth`)

Benchmarks D2 Lexical Support and D3 NLI Support against human-annotated hallucination spans from the RAGTruth dataset.

### Command to Run:
```powershell
.\rag-autopsy.bat ragtruth
```

### What This Demonstrates:
- On single-token entity substitutions (e.g., *"The treaty was signed in 1983"* instead of *"1783"*), **D2 Lexical Overlap achieves 0% recall** because 9 out of 10 tokens match the source context.
- **D3 NLI Cross-Encoder achieves 100% recall** because the cross-encoder detects the factual contradiction regardless of surface word overlap.
- **Output Artifact:** [`results/ragtruth/ragtruth_benchmark_report.json`](../results/ragtruth/ragtruth_benchmark_report.json).

---

## 11. Feature 10: Scalar RAG Quality vs. Stage Attribution (`rag-autopsy quality-compare`)

Directly compares conventional scalar evaluation frameworks (like Ragas) against causal stage attribution.

### Command to Run:
```powershell
.\rag-autopsy.bat quality-compare
```

### What This Demonstrates:
- Evaluates Ragas-style **Faithfulness** (claim entailment ratio) and **Context Precision@k** (mAP@k).
- Proves that while scalar quality metrics indicate *that* a response failed, they cannot differentiate between missing evidence, assembly loss, distractor corruption, or hallucination. Causal stage attribution is required for actionable engineering remediation.
- **Output Artifact:** [`results/rag_quality/quality_vs_attribution.json`](../results/rag_quality/quality_vs_attribution.json).

---

## 12. Feature 11: Full Regression & Boundary Leakage Test Suite (`pytest`)

Runs all automated unit, integration, determinism, and leakage tests.

### Command to Run:
```powershell
.venv\Scripts\pytest.exe -v
```

### What It Validates:
- **140 tests pass cleanly** in ~35 seconds.
- 24 dedicated negative tests in [`tests/test_trace_leakage.py`](../tests/test_trace_leakage.py) verify that attempting to inject privileged keys into `DiagnosticTrace` raises validation errors.
- Verifies bit-for-bit determinism, disk caching, BM25/Dense/Hybrid retrieval, and live Ollama Qwen 3 8B integration.
