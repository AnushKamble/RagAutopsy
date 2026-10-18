# RAG-Autopsy Preregistered & Frozen Evaluation Protocol

**Protocol Version:** 1.0.0  
**Registration Date:** 2026-10-18T23:30:00Z  
**Status:** FROZEN  
**Project Identifier:** AI-01 (RAG-Autopsy: Fault-Injection-Validated Failure Attribution for RAG Systems)  
**Configuration File:** [`configs/frozen_protocol_v1.yaml`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/configs/frozen_protocol_v1.yaml)  
**Frozen Calibration Parameters:** [`configs/frozen_calibration_params.yaml`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/configs/frozen_calibration_params.yaml)  

---

## 1. Research Questions & Formal Hypotheses

### RQ1: Black-Box Failure Attribution vs Simple Baselines
*Can low-cost observational diagnostics attribute observed RAG failures to the responsible pipeline stage without access to privileged evaluation metadata (gold evidence IDs, gold answers, or ground-truth injection labels)?*

- **Hypothesis H1 (Attribution Dominance):** The hierarchical diagnostic policy ($D_4$) achieves a Macro-F1 score whose 95% bootstrap confidence interval strictly exceeds the Majority-Cause Baseline and Random Guessing Baseline on held-out valid trials.
- **Hypothesis H2 (Structural Assembly Loss Identifiability):** Context assembly loss ($F_3$) is cleanly identifiable with $\ge 95\%$ recall from structural set differences between `pre_assembly_passage_ids` and `post_assembly_passage_ids` without accessing gold passage keys.

### RQ2: Upstream Retrieval Identifiability Under Varying Observability Regimes
*Can observational diagnostics disentangle evidence unavailability ($F_1$) from ranking displacement ($F_2$) under varying observability regimes without an expensive corpus oracle?*

- **Hypothesis H3a (Regime A Identifiability):** In Observability Regime A (Candidate Pool Trace with $K=20$), candidate pool score distribution and score-gap heuristics reliably disentangle $F_1$ from $F_2$ with $> 70\%$ recall.
- **Hypothesis H3b (Regime B Honest Abstention):** In Observability Regime B (Blind Top-$k$), missing evidence ($F_1$) and ranking displacement ($F_2$) are observationally indistinguishable. Diagnostics must abstain (`UNKNOWN: AMBIGUOUS_RETRIEVAL_MISS`) rather than faking diagnosability through unobservable assumptions.

### RQ3: Lexical Matching vs Semantic Entailment Grounding
*What is the precision and recall boundary of lexical token overlap ($D_2$) versus cross-encoder NLI ($D_3$) for grounding verification?*

- **Hypothesis H4 (Lexical Blind Spots):** Lexical token overlap ($D_2$) reliably detects verbatim extractions but suffers systematic blind spots:
  1. Valid abstractive paraphrases and synonym substitutions are falsely penalized as unsupported ($F_1$ false positive).
  2. Entity-copied predicate hallucinations ($F_5$) falsely register as grounded ($F_0$ false positive). Cross-encoder NLI ($D_3$) is required to resolve semantic entailment.

---

## 2. Source-Question Split Protocol & Grouping Invariants

To eliminate data leakage between diagnostic development, calibration, and final evaluation:

1. **Partitioning Unit:** The partition unit is strictly the **Source Question ID** (`question_id`).
2. **Grouping Invariant:** All perturbation variants ($F_0, F_1, F_2, F_3, F_4, F_5$) derived from a single source question $q_i$ must reside in the exact same partition. No question or associated documents may span multiple partitions.
3. **Partition Ratios:**
   - **Development Split (60%):** $N=60$ questions ($300$ potential trials). Used for heuristic exploration, rule implementation, and unit testing.
   - **Calibration Split (20%):** $N=20$ questions ($100$ potential trials). Used solely for tuning thresholds ($\tau_{\text{low}}, \tau_{\text{gap}}, \tau_{\text{overlap}}$).
   - **Held-Out Test Split (20%):** $N=20$ questions ($100$ potential trials). **Completely quarantined and untouched** during all tuning and iterative development.
4. **Disjointness Invariant:**
   $$\text{IDs}(\text{dev}) \cap \text{IDs}(\text{calibration}) = \emptyset$$
   $$\text{IDs}(\text{dev}) \cap \text{IDs}(\text{test}) = \emptyset$$
   $$\text{IDs}(\text{calibration}) \cap \text{IDs}(\text{test}) = \emptyset$$
   Enforced deterministically by [`SplitManifest`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/src/rag_autopsy/data/splits.py#L13-L60) and frozen in [`configs/split_manifest_100.json`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/configs/split_manifest_100.json).

---

## 3. Fault Parameters, Perturbation Severity, & Injection Contracts

| Fault Code | Failure Mode | Pipeline Stage | Injection Mechanism & Severity | Preconditions | State Transition Verification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$F_0$** | Nominal Baseline | None | None (clean execution, $\alpha=0.0$). | Corpus contains gold evidence. | Nominal execution, zero pipeline interventions. |
| **$F_1$** | Evidence Unavailable | Knowledge Base / Corpus | Complete purge ($\alpha=1.0$) of gold evidence passages from trial-scoped index. | Gold evidence passage exists in corpus prior to trial. | Post-injection index contains 0 gold passages. |
| **$F_2$** | Evidence Ranked Out | Retriever / Ranker | $N_{\text{distractors}}=5$ query-targeted distractors injected with BM25 boost $\beta=1.5$. Cutoff $k=3$. | Gold evidence passage exists in corpus. | Gold passage searchable in candidate pool, but rank $> 3$. |
| **$F_3$** | Context Assembly Loss | Prompt Context Builder | Gold passage dropped during prompt assembly (context budget truncation simulation, $\gamma=1.0$). | Gold passage retrieved in top-$k$ ($k \le 3$). | Gold passage present in `pre_assembly_passage_ids` but omitted from `post_assembly_passage_ids`. |
| **$F_4$** | Contradictory Evidence | Context Integration | Antonym and negation mutation injected into context passage, conflicting with gold claim. | Gold passage retrieved in top-$k$. | Both original fact and contradictory fact present in `generator_context`. |
| **$F_5$** | Unsupported Generation | Generator / LLM | Generator forced to substitute entities/predicates, fabricating an unsupported claim. | Gold evidence present in context. | Generated answer asserts predicate unsupported or refuted by context. |

---

## 4. Injector Validity Criteria & Exclusion Logic

A trial is certified as valid if and only if:
$$\text{trial\_valid} = \text{injection\_applied} \land \text{trace\_condition\_achieved} \land \text{preconditions\_satisfied}$$

### Validity Adjudication Table
- **Attempted Trial:** Every instantiated pipeline run targeting a fault intervention.
- **Valid Trial:** An intervention that verified all physical preconditions and achieved its pre-declared execution trace signature.
- **Invalid Trial:** An intervention where preconditions failed (e.g. gold passage was missed naturally by retriever in $F_3$, meaning assembly loss cannot be isolated).

| Invalid Reason Code | Cause & Exclusion Justification |
| :--- | :--- |
| `NO_GOLD_EVIDENCE_IN_CORPUS` | Source QA item lacked valid gold evidence passages in base corpus. |
| `GOLD_STILL_IN_TOP_K_DESPITE_DISTRACTORS` | Distractor injection failed to push gold passage beyond top-$k$. |
| `PRECONDITION_FAILED_GOLD_MISSED_TOP_K` | Retriever missed gold passage naturally; downstream stages ($F_3, F_4$) cannot be evaluated in isolation. |
| `CONTRADICTION_NOT_ASSEMBLED_INTO_CONTEXT` | Injected contradiction passage was dropped prior to generator context. |

*Protocol Exclusion Rule:* All attempted trials are recorded in the machine-readable validity accounting report. Invalid trials are strictly excluded from primary attribution metrics to prevent unverified injections from polluting diagnostic benchmarks.

---

## 5. Diagnostic-Visible Features vs Forbidden Privileged Fields

### Permitted Observational Trace Features ([`DiagnosticTrace`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/src/rag_autopsy/core/schemas.py#L72-L108))
1. `trial_id` (anonymized hash identifier)
2. `question_id` (source question identifier)
3. `question` (raw query string)
4. `retrieved_top_k` (list of passages: `passage_id`, `rank`, `score`, `text`)
5. `candidate_pool_scores` (float array of candidate pool BM25 scores; available only in Regime A)
6. `source_passage_ids` (passage IDs returned by retriever)
7. `pre_assembly_passage_ids` (passage IDs delivered to context builder)
8. `post_assembly_passage_ids` (passage IDs successfully packed into prompt context)
9. `generator_context` (complete prompt context string presented to generator)
10. `answer` (generated response text)
11. `latency_ms` (execution timing)
12. `generator_model`, `generator_config` (model architecture metadata)

### Forbidden Privileged Fields ([`EvaluatorRecord`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/src/rag_autopsy/core/schemas.py#L111-L135))
The following fields are strictly quarantined in [`EvaluatorStore`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/src/rag_autopsy/core/schemas.py#L138-L165) and **never** visible to diagnostics:
- ❌ `injected_fault` (ground-truth cause)
- ❌ `fault_parameters` (injection parameters, perturbation rates)
- ❌ `gold_answer` / `gold_answers` (reference answers)
- ❌ `gold_evidence_ids` (gold document identifiers)
- ❌ `target_rank` (unobserved gold passage rank in candidate pool)
- ❌ `injection_applied`
- ❌ `trace_condition_achieved`
- ❌ `trial_valid`
- ❌ `invalid_reason`

Any diagnostic attempting to ingest an [`EvaluatorRecord`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/src/rag_autopsy/core/schemas.py#L111-L135) or access forbidden keys will trigger an immediate, fatal `TypeError: Data boundary violation`.

---

## 6. Threshold Selection & Calibration Protocol

All diagnostic thresholds were tuned on the **20% Calibration Partition** ($N=20$ questions, $100$ trials) using grid search maximizing balanced accuracy on non-abstained trials. 

**Frozen Parameters ([`configs/frozen_calibration_params.yaml`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/configs/frozen_calibration_params.yaml)):**
- $\tau_{\text{low}} = 2.0$ (`low_score_threshold` for $D_1$)
- $\tau_{\text{gap}} = 1.0$ (`score_gap_threshold` for $D_1$)
- $\tau_{\text{overlap}} = 0.30$ (`min_overlap_ratio` for $D_2$)
- $\tau_{\text{conf}} = 0.65$ (`confidence_threshold` for $D_4$ abstention)

*Invariant:* These values are frozen and will not be re-tuned on the held-out test split.

---

## 7. Primary & Secondary Evaluation Metrics

### Primary Metrics
1. **Macro-Averaged F1 ($F_1^{\text{macro}}$):** Unweighted mean of per-class F1 scores across all target classes. Abstentions are counted as errors (`UNMAPPED_OR_UNKNOWN`).
2. **Balanced Accuracy ($\text{BA}$):** Macro-averaged recall across all classes:
   $$\text{BA} = \frac{1}{|C|} \sum_{c \in C} \text{Recall}_c$$
3. **Coverage ($C$):** Proportion of trials where the diagnostic issued a definitive prediction:
   $$C = \frac{N_{\text{covered}}}{N_{\text{total}}} = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i \ne \text{UNKNOWN})}{N}$$
4. **Conditional Accuracy ($A_{\text{cov}}$):** Accuracy computed strictly on covered trials:
   $$A_{\text{cov}} = \frac{1}{N_{\text{covered}}} \sum_{i \in \text{Covered}} \mathbb{I}(\hat{y}_i = y_i)$$

### Secondary Metrics
1. **Per-Class Precision, Recall, F1, Support:** Evaluated for each fault class $c \in \{F_0, F_1, F_2, F_3, F_4, F_5\}$.
2. **Raw and Row-Normalized Confusion Matrices:** Showing exact transition counts and conditional probabilities $P(\hat{y} \mid y)$.
3. **Diagnostic Runtime:** Total latency (ms) and average latency per trial (ms/trial).

---

## 8. Clustered Bootstrap Resampling Protocol

- **Resampling Unit:** Clustered non-parametric bootstrap strictly at the **Question ID** (`question_id`) level.
- **Scientific Justification:** Because all 5-6 fault variants of a question share the same base passage context, standard IID trial-level bootstrapping would artificially inflate degrees of freedom and underestimate confidence interval widths. Resampling question clusters preserves intra-cluster correlation.
- **Iterations ($B$):** $1,000$ bootstrap replicates.
- **Significance Level ($\alpha$):** $0.05$ (two-tailed $95\%$ percentile confidence interval $[2.5\%, 97.5\%]$).
- **Random Seed:** Fixed seed $42$.

---

## 9. Abstention Protocol

When observational features cannot reliably disambiguate competing failure causes:
1. **Regime B Retrieval Ambiguity:** When top-score $< \tau_{\text{low}}$ and candidate pool scores are inaccessible, diagnostics must output `UNKNOWN: AMBIGUOUS_RETRIEVAL_MISS`.
2. **Conflicting Signals in Policy Engine:** When $D_1$ indicates clean retrieval ($F_0$) but $D_2$ indicates missing lexical support without explicit contradiction, and confidence $< 0.65$, $D_4$ must output `UNKNOWN`.
3. **Scoring:** Abstained trials are excluded when computing conditional accuracy $A_{\text{cov}}$, but penalized as misclassifications in overall macro-F1 and balanced accuracy.

---

## 10. Planned Baselines and Ablations

### Baselines
1. **Majority-Cause Baseline:** Unconditionally predicts the empirical majority class from the calibration split ($F_0$).
2. **Random Guessing Baseline:** Uniform random choice over $\{F_0, F_1, F_2, F_3, F_4\}$ (`seed=42`).

### Diagnostics Under Test
1. **$D_1$ Standalone Retrieval Diagnostic:** Evaluates score thresholds, score cliffs, and structural assembly diffs.
2. **$D_2$ Standalone Lexical Support Diagnostic:** Evaluates token overlap and contradiction cues.
3. **$D_3$ Standalone NLI Support Diagnostic:** Cross-encoder NLI entailment / contradiction scoring.
4. **$D_4$ Combined Attribution Policy:** Multi-signal hierarchical policy with calibrated abstention.
5. **$D_4$ Learned Classifier Policy:** Decision tree / logistic regression trained on dev+calibration feature vectors.

### Planned Ablations
1. **Regime A vs Regime B:** Evaluating the impact of candidate pool visibility on $F_1$ vs $F_2$ identifiability.
2. **Abstention Enabled vs Disabled:** Forced-choice evaluation versus calibrated abstention.
3. **Heuristic Precedence vs Learned Feature Weights:** Comparing deterministic decision cascades against learned decision boundaries.
