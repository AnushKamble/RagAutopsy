# RAG-Autopsy Natural Failure Annotation Guide

This guide defines the standardized review protocol for auditing unmodified (natural) failures in the RAG pipeline.

---

## 1. Scope and Objective
Controlled fault injection establishes internal attribution validity under known interventions. The natural failure audit tests **external transfer**: how well the diagnostic rules and observed signatures transfer to organic failures that occur without artificial perturbation.

Reviewers must follow this rubric strictly. Do not guess; use `UNKNOWN` or `MULTIPLE_PLAUSIBLE` when evidence is ambiguous.

---

## 2. Review Workflow and Questions

For each qualifying unmodified failure (where `answer_exact_match == False` under clean F0 pipeline execution), the reviewer evaluates six dimensions:

### Q1: Answer Correctness
- **YES:** The answer is factually and semantically equivalent to the reference answer despite formatting differences.
- **NO:** The answer is factually incorrect, incomplete, or fails to answer the question.
- **PARTIALLY:** The answer contains partial truth or answers only one hop of a multi-hop question.

### Q2: Context Grounding / Support
- **SUPPORTED:** Every factual claim in the generated answer is directly substantiated by the exact text in `generator_context`.
- **UNSUPPORTED:** The answer makes material assertions not found in `generator_context` (hallucination or parametric memory leakage).
- **CONTRADICTED:** The answer contradicts facts explicitly stated in `generator_context`.

### Q3: Evidence Retrieval Status
- **RETRIEVED_FULL:** All necessary supporting passages are present in `retrieved_top_k`.
- **RETRIEVED_PARTIAL:** At least one, but not all, supporting passages are in `retrieved_top_k`.
- **MISSED:** None of the necessary supporting passages are in `retrieved_top_k`.

### Q4: Context Assembly Survival
- **SURVIVED:** All retrieved supporting passages survived into `post_assembly_passage_ids`.
- **TRUNCATED:** One or more retrieved supporting passages were omitted due to token/character limits or formatting drops.
- **NOT_APPLICABLE:** Evidence was never retrieved in the first place.

### Q5: Context Contradiction
- **YES:** The generator context contains contradictory statements regarding key entities.
- **NO:** Context passages are mutually consistent.

### Q6: Adjudicated Primary Failure Cause
- **F1_UNAVAILABLE:** Evidence was absent from the candidate pool or index.
- **F2_RANKED_OUT:** Evidence was in the corpus, but BM25 scored distractors higher, pushing evidence below top-$k$.
- **F3_ASSEMBLY_LOSS:** Evidence was retrieved in top-$k$, but truncated/omitted during prompt assembly.
- **F4_CONTRADICTION:** Conflicting statements in context confused the generator.
- **REASONING_ERROR:** Adequate evidence was retrieved and present in context, but the generator failed to perform multi-hop reasoning or synthesis correctly.
- **MULTIPLE_PLAUSIBLE:** Multiple stages contributed to the failure (e.g. partial retrieval combined with generator reasoning failure).
- **UNKNOWN:** Observability traces are insufficient to determine cause.

---

## 3. Disagreement and Adjudication Protocol
If two independent reviewers label a trial:
- Cohen's Kappa ($\kappa$) is reported on the discrete cause classification.
- In case of disagreement, an adjudicator reviews both rationale notes to establish the final consensus label.
- If only one reviewer is available, a delayed second pass (minimum 48 hours) is conducted as an **intra-rater consistency check**, not inter-annotator agreement.
