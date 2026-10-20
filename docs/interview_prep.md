# RAG-Autopsy: Technical Interview Preparation & Defense Guide

This guide provides deep-dive interview questions, architectural explanations, and defensible answers for discussing RAG-Autopsy in machine learning systems, research, and applied AI interviews.

---

## 1. Architectural & Causal Design Questions

### Q1: What is the core problem RAG-Autopsy solves, and how does it differ from RAG evaluation frameworks like Ragas or TruLens?
**Candidate Answer Outline:**
- **Problem Formulation:** Existing RAG evaluators (e.g., Ragas, TruLens, ARES) compute aggregate scalar quality scores (e.g., "Faithfulness: 0.65", "Context Relevance: 0.70") using expensive LLM-as-a-judge prompts. While they indicate *that* a system failed, they do not pinpoint *which architectural stage caused* the failure without privileged gold metadata or expensive re-querying.
- **RAG-Autopsy Difference:** RAG-Autopsy treats failure attribution as a causal diagnosis problem under strict runtime observability boundaries:
  1. We introduce a controlled, state-verified fault-injection taxonomy spanning the pipeline:
     - $F1$: Corpus evidence unavailability
     - $F2$: Retriever rank shortfall
     - $F3$: Context window truncation / assembly drop
     - $F4$: Contradictory distractor evidence
     - $F5$: Generator hallucination / unsupported generation
     - $F0$: Clean control execution
  2. We develop low-cost ($<1$ ms latency), rule-based and learned diagnostics operating *strictly* on unprivileged runtime traces (passage scores, token overlap, pre/post context IDs) without access to gold labels or hidden corpus metadata.
  3. We evaluate attribution accuracy using pre-registered metrics, question-clustered bootstrap uncertainty, and transfer testing on unmodified natural failures.

---

### Q2: How did you mathematically resolve the identifiability problem between F1 (Unavailable) and F2 (Ranked Out)?
**Candidate Answer Outline:**
- **Identifiability Conflict:** In standard RAG pipelines observing only top-$k$ returned passages (e.g., $k=3$), a missing gold passage presents an identical trace signature whether the document is completely missing from the corpus ($F1$) or ranked at position $k+1=4$ ($F2$). Diagnosing $F2$ over $F1$ from top-$k$ text alone is mathematically impossible without privileged knowledge.
- **Two Pre-Registered Regimes:**
  - **Regime A (Candidate Pool Observability):** The retriever exposes summary statistics from a broader candidate pool (top-$K$, $K=20$). If the top-1 score is low and candidate scores drop smoothly, evidence is likely unavailable ($F1$). If candidate scores contain a steep drop ("score cliff") or high scores exist outside the top-$k$ window, it signals a rank shortfall ($F2$).
  - **Regime B (Blind Top-$k$ Observability):** Only top-$k$ passages and their scores are logged. Diagnosers cannot separate $F1$ and $F2$. Rather than fitting spurious dataset artifacts or cheating with gold labels, our combined policy **honestly abstains** with `UNKNOWN: AMBIGUOUS_RETRIEVAL_MISS` or collapses them into a unified retrieval miss class.
- **Empirical Impact:** In Regime A, D4 achieves **0.701 Macro-F1** (5-class). In Regime B, Macro-F1 drops by 23.3 percentage points to **0.468**, demonstrating that candidate pool observability is a necessary precondition for sub-stage retrieval attribution.

---

### Q3: How did you guarantee zero data leakage between the evaluator and the diagnostic system?
**Candidate Answer Outline:**
- **Pydantic Schema Isolation:**
  - [`DiagnosticTrace`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/src/rag_autopsy/core/schemas.py): Contains strictly runtime-observable fields: `query`, `retrieved_passages` (ID, rank, score, text), `pre_assembly_passage_ids`, `post_assembly_passage_ids`, `context_text`, `generated_answer`, and `execution_latency_ms`. Model is configured with `extra="forbid"`.
  - [`EvaluatorRecord`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/src/rag_autopsy/core/schemas.py): Contains privileged fields: `injected_fault`, `injection_parameters`, `gold_answer`, `gold_evidence_ids`, `is_valid`, and `adjudication_reason`.
- **Structural Separation Mechanisms:**
  1. **Physical Storage:** Traces and evaluation records are serialized to separate JSONL files (`diagnostic_traces.jsonl` vs `evaluator_records.jsonl`).
  2. **Opaque Trial IDs:** Trial identifiers are generated as deterministic 16-hex hashes (`trial_0a1b2c3d4e5f6789`) seeded by question ID and random salt. No fault tokens (`f1`, `f2`) appear in filenames, directory paths, or trial IDs.
  3. **Presentation Shuffling:** Traces are delivered to diagnostics in a randomized presentation order independent of the generator execution order.
  4. **Synthetic Passage Masking:** When injecting synthetic contradictors or distractors, passage IDs are masked using opaque SHA-256 hashes (`passage_syn_{hash}`), and document titles are replaced with neutral synthetic headers to prevent keyword snooping.
  5. **Negative Test Suite:** 24 dedicated pytest leakage tests in [`tests/test_trace_leakage.py`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/tests/test_trace_leakage.py) verify that injecting evaluator keys into `DiagnosticTrace` raises validation errors, and diagnostic function signatures accept only `DiagnosticTrace`.

---

### Q4: Why did you use Clustered Bootstrap Resampling instead of standard i.i.d. bootstrap?
**Candidate Answer Outline:**
- **Cluster Dependency:** In our evaluation setup, multiple intervention trials (e.g., $F0, F1, F2, F3, F4, F5$) are generated from the *same* underlying source question.
- **Underestimation of Variance:** If standard i.i.d. bootstrap resampling was performed at the trial level, trials from the same question would appear across both resampled folds, violating the independence assumption. Questions with idiosyncratic lexical properties (e.g., unusually specific named entities) would artificially reduce empirical variance and produce overconfident confidence intervals.
- **Clustered Resampling Unit:** In [`src/rag_autopsy/evaluation/bootstrap.py`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/src/rag_autopsy/evaluation/bootstrap.py), the cluster unit is strictly the `question_id`. During each bootstrap iteration ($B=500$ or $B=1,000$), entire question clusters are sampled with replacement, preserving intra-question correlation structure and yielding valid, conservative 95% confidence intervals.

---

### Q5: Why did your diagnostic system achieve 0.0% transfer agreement on natural (unmodified) HotpotQA failures?
**Candidate Answer Outline:**
- **The Empirical Finding:** When evaluating 100 clean, unmodified HotpotQA development questions, human adjudication revealed:
  - 90% failed due to generator **reasoning errors** on complex multi-hop synthesis.
  - 9% failed due to retrieval rank shortfall ($F2$).
  - 1% failed due to missing corpus evidence ($F1$).
  - The pipeline transit diagnostics achieved **0.0% agreement** on the natural failure cause distribution.
- **Root Cause Analysis:**
  - Our controlled fault injection framework tests **pipeline transit and integrity faults** (e.g., dropped passages, score drops, surface contradictions).
  - In a clean, high-performing pipeline, however, BM25 successfully retrieves the gold passages into context for ~90% of questions. The bottleneck is the generator failing to perform deductive multi-hop inference over valid evidence.
  - Transit diagnostics inspect *whether* evidence arrived and *whether* the answer lexically/semantically grounds in context. They cannot detect subtle multi-hop reasoning fallacies without a specialized deductive reasoning judge.
- **Scientific Value:** Rather than concealing this negative finding or claiming universal diagnostic coverage, we documented it as an ADR and core paper contribution: *diagnostics calibrated on synthetic transit faults cannot diagnose generator reasoning bottlenecks without semantic reasoning evaluators.*

---

## 2. Quantitative Results & Metric Definitions

### Primary Benchmark Metrics (Held-Out Test Partition, 6-Class, N=114 Valid Trials)

| Metric | Baseline Majority | Baseline Random | D1 Retrieval | D2 Lexical | D3 NLI | D4 Heuristic | D4 Learned (Decision Tree) |
|---|---|---|---|---|---|---|---|
| **Macro-F1** | 0.0498 | 0.2426 | 0.3789 | 0.3852 | 0.3743 | 0.7181 | **0.7421** |
| **95% Bootstrap CI** | [0.0476, 0.0533] | [0.1716, 0.3191] | [0.3494, 0.4028] | [0.3415, 0.4167] | [0.3190, 0.4228] | [0.6568, 0.7652] | **[0.6865, 0.7778]** |
| **Balanced Acc** | 0.1667 | 0.2454 | 0.4833 | 0.4741 | 0.4472 | 0.7611 | **0.7981** |
| **Coverage** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 97.37% | **100.0%** |
| **Accuracy (Covered)**| 17.54% | 24.56% | 49.12% | 46.49% | 43.86% | 77.48% | **78.95%** |

### Per-Class Performance Breakdown (D4 Learned Decision Tree)

| Fault Class | Precision | Recall | F1-Score | Support ($N_{valid}$) | Observable Diagnostic Mechanism |
|---|---|---|---|---|---|
| **F0 (Clean Control)** | 48.65% | 90.00% | 0.6316 | 20 | Normal retrieval scores, high lexical/NLI support, EM match |
| **F1 (Unavailable)** | 0.00% | 0.00% | 0.0000 | 20 | Unidentifiable without gold relevance (retrieval pool uniformly low) |
| **F2 (Ranked Out)** | 100.00% | 100.00% | **1.0000** | 20 | Candidate pool score cliff: high sub-top-$k$ candidate score |
| **F3 (Assembly Loss)**| 78.26% | 100.00% | **0.8780** | 18 | Set difference: $\text{PreAssemblyIDs} \setminus \text{PostAssemblyIDs} \neq \emptyset$ |
| **F4 (Contradiction)**| 100.00% | 94.44% | **0.9714** | 18 | Context polarity conflict: lexical negation / NLI contradiction signal |
| **F5 (Hallucination)**| 100.00% | 94.44% | **0.9714** | 18 | Support deficit: low lexical overlap and low NLI entailment score |

---

## 3. Failure Counterexamples & Blind Spots

### Counterexample 1: D1 False Positive on Low-IDF Queries (Score Cliff False Alarm)
- **Scenario:** A broad, generic question (e.g., *"What is the capital and largest city of the state?"*) produces naturally flat BM25 candidate scores with low term rarity.
- **Diagnostic Behavior:** D1's calibrated score-gap threshold flags the drop between candidate 3 and candidate 4 as an $F2$ rank-out event, even though the true evidence is already in top-3.
- **Remediation:** Calibrate score-drop thresholds relative to the candidate pool variance ($\sigma_{scores}$) rather than absolute score margins.

### Counterexample 2: D2 False Positive on Paraphrased Generation
- **Scenario:** The generator produces a semantically accurate answer using high-level synonyms or abstractive phrasing rather than copying the exact extractive span from the context.
- **Diagnostic Behavior:** D2 measures token overlap (Jaccard similarity / token containment). The overlap score drops below the $\tau_{support} = 0.40$ threshold, triggering an erroneous $F5$ (Unsupported Generation) diagnosis on an accurate answer.
- **Remediation:** Combine lexical overlap with D3 NLI cross-encoder scoring, or compute token embeddings (e.g., BERTScore) to capture semantic equivalence.

### Counterexample 3: D2 False Negative on Entity Substitution (High Lexical Overlap)
- **Scenario:** The generator outputs: *"The Treaty of Paris was signed in 1983."* instead of *"1783."*
- **Diagnostic Behavior:** Because 9 out of 10 tokens match the source context exactly, D2 computes a 90% lexical support score and diagnoses $F0$ (Clean), missing the hallucination completely.
- **Empirical Proof:** On the RAGTruth benchmark ([`src/rag_autopsy/evaluation/ragtruth_eval.py`](file:///C:/Users/anush/OneDrive/Desktop/Project/Projects/AIML/RagAutopsy/src/rag_autopsy/evaluation/ragtruth_eval.py)), D2 achieves 0% recall on single-token factual substitutions, whereas D3 NLI achieves 100% recall.

### Counterexample 4: Universal Blindness to F1 (Unavailable Evidence)
- **Scenario:** A user asks a question about an entity completely absent from the indexing corpus.
- **Diagnostic Behavior:** The retriever returns the 3 highest scoring passages in the corpus (which are irrelevant distractors with low BM25 scores). The candidate pool scores are also uniformly low.
- **Identifiability Barrier:** Without access to a gold relevance label or external world knowledge, a diagnostic cannot tell whether the query is simply difficult (low similarity scores for all documents) or if the evidence document does not exist in the collection. All diagnostics achieve 0% recall on $F1$.
