# RAG-Autopsy Frozen Evaluation Protocol

**Protocol Version:** 1.0.0
**Status:** FROZEN
**Date:** 2026-10-18

---

## 1. Research Question and Hypotheses

### Primary Question
How accurately can low-cost diagnostics attribute observed RAG failures to the pipeline stage or intervention responsible, compared with simple baselines, on a held-out evaluation sample?

### Pre-declared Hypotheses
- **H1 (Attribution vs Baseline):** The combined policy (D4) achieves a Macro-F1 whose 95% bootstrap CI strictly exceeds the Majority-Cause Baseline on valid held-out injected trials.
- **H2 (Identifiability Boundary):** In Regime A (Candidate Pool Trace with $K=20$), D1/D4 distinguishes F1 (evidence absent) from F2 (ranked out) using score-cliff heuristics; in Regime B (Blind Top-$k$), F1 and F2 are indistinguishable, necessitating abstention (`UNKNOWN: AMBIGUOUS_RETRIEVAL_MISS`).
- **H3 (Assembly Loss Separability):** Context assembly loss (F3) is cleanly identifiable from structural diffs between pre-assembly and post-assembly traces without access to gold evidence keys.

---

## 2. Dataset and Partitioning

- **Source Corpus:** HotpotQA Development Split (distractor setting, `v1.1`, CC BY-SA 4.0).
- **Subset Rule:** Deterministic subset of $N=100$ questions ($1,000$ passages) sampled with fixed seed (`seed=42`).
- **Partitioning Ratios:**
  - 60% Development ($N=60$ questions) — Diagnostic rule authoring and unit testing.
  - 20% Calibration ($N=20$ questions) — Threshold tuning (`low_score_threshold`, `score_gap_threshold`, `min_overlap_ratio`).
  - 20% Held-Out Test ($N=20$ questions) — Final evaluation. **Untouched during development and tuning.**
- **Grouped Disjointness Invariant:**
  All perturbations (F0, F1, F2, F3, F4) of a single question reside in the same partition:
  $$\text{IDs}(\text{dev}) \cap \text{IDs}(\text{calibration}) \cap \text{IDs}(\text{test}) = \emptyset$$

---

## 3. Intervention Contracts and Validity Adjudication

For each trial, the intervention must achieve its pre-declared physical trace condition to be counted as a valid trial:

| Fault | Intervention Contract | Precondition | Verification Rule | Invalid Reason Code |
|---|---|---|---|---|
| **F0** | Default clean execution | Gold passage in corpus | Trace condition achieved | None |
| **F1** | Trial-scoped corpus without gold passages | Gold passages identified | Gold passages completely absent from trial corpus | `Gold passages already absent from base corpus` |
| **F2** | 5 query-targeted distractors added to corpus | Gold passages identified | Gold passage searchable in corpus, but rank $> k$ (cutoff $k=3$) | `Gold passage still ranked in top-k despite distractors` |
| **F3** | Drop gold passage from retrieved list | Gold passage retrieved in top-$k$ | Gold in `pre_assembly_ids`, absent from `post_assembly_ids` | `Precondition failed: No gold passage was retrieved in top-k` |
| **F4** | Injected contradictory passage into context | Gold passage retrieved in top-$k$ | Both original evidence and contradiction exist in context | `Precondition failed: Original gold evidence was not retrieved` |

**Adjudication Rule:** Ineffective or invalid injections are recorded in the validity report but excluded from primary attribution metrics.

---

## 4. Diagnostics Under Evaluation

1. **Majority-Cause Baseline:** Predicts the most frequent class in the calibration partition.
2. **Random Baseline:** Uniform random choice over $\{F0, F1, F2, F3, F4\}$ (`seed=42`).
3. **D1 Retrieval Diagnostic:** Evaluates top-$k$ score, candidate pool score drop-off, and pre/post structural diffs.
4. **D2 Lexical Support Diagnostic:** Evaluates answer-context keyword overlap and scans for contradiction cues.
5. **D4 Combined Attribution Policy:** Hierarchical synthesis with explicit abstention (`UNKNOWN`) on conflicting signals.

---

## 5. Statistical and Reporting Protocol

- **Primary Metric:** Macro-Averaged F1 across all target classes.
- **Secondary Metrics:** Balanced Accuracy, Per-Class Precision / Recall / F1, Overall Accuracy, Coverage, Conditional Accuracy.
- **Confusion Matrices:** Raw counts and true-cause normalized (row sums = 1.0).
- **Uncertainty Estimation:** Non-parametric clustered bootstrap resampling strictly at the **Question ID level** with $B=1,000$ iterations ($\alpha=0.05$, 95% percentile confidence interval).
- **Output Artifacts:**
  - Machine-readable JSON summary: `results/evaluation_summary.json`
  - Per-trial predictions CSV: `results/trial_predictions.csv`
