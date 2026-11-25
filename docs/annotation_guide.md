# RAG-Autopsy Natural Failure Reviewer Annotation Guide & Protocol

**Document Version:** 2.0.0
**Status:** FROZEN
**Target:** Auditing unmodified (organic) RAG failures and evaluating diagnostic transfer
**Related Specifications:** [`docs/preregistered_evaluation_protocol.md`](preregistered_evaluation_protocol.md), [`docs/taxonomy.md`](taxonomy.md), [`docs/limitations.md`](limitations.md)

---

## 1. Scope and Anti-Leakage Boundary

Controlled fault injection ($F_1 \dots F_5$) establishes **internal attribution validity** under known, verified interventions. The natural failure audit tests **external transfer**: determining whether diagnostics designed for injected transit faults remain useful when applied to organic errors generated without artificial perturbations.

### Core Methodological Invariants
1. **Zero Fault Injection:** The audit sample must consist strictly of unmodified, clean pipeline runs (`injected_fault == "F0"`, `injection_applied == False`). Artificially perturbing this sample and labeling it "natural" is strictly forbidden.
2. **Frozen Settings:** The retrieval backend (BM25, $k=3, K_{\text{pool}}=20$), context builder (3,000 char limit), generator configuration, and diagnostic thresholds ($\tau_{\text{low}}=2.0, \tau_{\text{gap}}=1.0, \tau_{\text{overlap}}=0.30$) must remain frozen as pre-registered in [`configs/frozen_calibration_params.yaml`](../configs/frozen_calibration_params.yaml).
3. **No Retuning on Audit Data:** Reviewer labels from this audit must **never** be used to recalibrate diagnostic thresholds or modify decision policies. The audit set must remain genuinely held out.
4. **Honest Sample Accounting:** If too few natural failures occur under clean execution, the exact count and statistical limitations must be documented. Reviewers must never manufacture additional failures to reach an arbitrary quota.

---

## 2. Sampling Rule & Case Selection Protocol

Natural failures are identified via an explicit deterministic sampling procedure:

1. **Source Pool:** Unperturbed clean execution ($F_0$) over the designated held-out evaluation questions (e.g. the 20 questions in the held-out test split, or the full 100-question evaluation pool).
2. **Qualification Criterion:** An execution qualifies as a natural failure if the clean pipeline produces an objectively incorrect answer under normalized comparison:
   $$\text{Natural Failure} \iff \text{answer\_exact\_match} == \text{False} \lor \text{token\_f1} < 1.0$$
3. **Trace Preservation:** For every qualifying natural failure, the complete operational execution trace ([`DiagnosticTrace`](../src/rag_autopsy/core/schemas.py)) and evaluator record ([`EvaluatorRecord`](../src/rag_autopsy/core/schemas.py)) are archived for independent adjudication.

---

## 3. The 8-Dimensional Reviewer Rubric

Reviewers evaluate each qualifying failure across 8 standardized dimensions. Ratings must be grounded solely in the observable text and documented evidence:

```
                                  [Natural Failure Trace]
                                             |
            +--------------------------------+--------------------------------+
            |                                |                                |
    [Q1: Correctness]               [Q2: Context Grounding]          [Q3: Evidence Retrieval]
  (CORRECT / PARTIAL /             (SUPPORTED / PARTIAL /             (FULL / PARTIAL /
      INCORRECT)                   UNSUPPORTED / CONFLICT)                 MISSED)
            |                                |                                |
            +--------------------------------+--------------------------------+
            |                                |                                |
   [Q4: Context Survival]          [Q5: Context Contradiction]      [Q6: Likely Primary Cause]
   (SURVIVED / TRUNCATED /          (CONTRADICTION_PRESENT /          (REASONING_ERROR / F2 /
            N/A)                          CONSISTENT)                   F1 / F3 / F4)
            |                                |                                |
            +--------------------------------+--------------------------------+
            |                                                                 |
   [Q7: Secondary Causes]                                             [Q8: Unknown Flag]
   (Multiple Plausible Factors)                                       (Insufficient Evidence)
```

### Dimension 1: Answer Correctness (`answer_correctness`)
- **`CORRECT`:** The answer is semantically and factually equivalent to the gold reference answer, differing only in punctuation, capitalization, or minor phrasing (e.g., `"48,982"` vs `"48982"`).
- **`PARTIALLY_CORRECT`:** The answer captures part of the required multi-hop relation or answers one entity in a comparison query, but omits critical components (e.g., naming one of two opera composers).
- **`INCORRECT`:** The answer is factually wrong, provides an unrelated entity, or is an extraction failure (e.g., extracting a section header instead of the entity).

### Dimension 2: Support in Exact Generator Context (`support_in_context`)
Evaluates whether material claims made in the generated answer are grounded in the exact string `generator_context`:
- **`SUPPORTED`:** Every claim in the generated answer is directly asserted by the retrieved passages present in `generator_context`.
- **`PARTIALLY_SUPPORTED`:** Some components of the answer are grounded in context, but other claims are ungrounded.
- **`UNSUPPORTED`:** The generated answer asserts facts that do not appear anywhere in `generator_context` (hallucination or parametric memory leakage).
- **`CONTRADICTED`:** The generated answer asserts a claim that is directly refuted by facts stated in `generator_context`.

### Dimension 3: Evidence Retrieval Status (`evidence_retrieval`)
Compares the retriever's top-$k$ output (`pre_assembly_passage_ids`) against the gold evidence set $\{p_{\text{gold}}\}$:
- **`RETRIEVED_FULL`:** All required gold evidence passages are present in `retrieved_top_k` ($| \{p_{\text{gold}}\} \cap \text{top\_k} | = |\{p_{\text{gold}}\}|$).
- **`RETRIEVED_PARTIAL`:** At least one, but not all, required gold evidence passages are present in `top_k`. (Common in multi-hop queries where one hop is retrieved and the second is ranked out).
- **`MISSED`:** None of the required gold evidence passages are present in `retrieved_top_k` ($| \{p_{\text{gold}}\} \cap \text{top\_k} | = 0$).

### Dimension 4: Context Assembly Survival (`context_survival`)
Verifies whether retrieved gold evidence survived the context builder prompt assembly:
- **`SURVIVED`:** All gold passages present in `pre_assembly_passage_ids` appear in `post_assembly_passage_ids`.
- **`TRUNCATED`:** One or more retrieved gold passages were omitted during prompt construction due to token/character limits or sorting drops.
- **`NOT_APPLICABLE`:** No gold evidence was retrieved in top-$k$ to begin with.

### Dimension 5: Context Contradiction (`context_contradiction`)
- **`CONTRADICTION_PRESENT`:** The assembled `generator_context` contains conflicting claims regarding key query entities (e.g. contradictory dates or conflicting biographical facts).
- **`CONSISTENT`:** Passages in context are mutually consistent.

### Dimension 6: Likely Primary Failure Cause (`likely_primary_cause`)
The single primary bottleneck identified by the reviewer following the decision hierarchy:
1. **`F1_UNAVAILABLE`:** Gold evidence is completely absent from the searchable candidate pool ($K_{\text{pool}}=20$).
2. **`F2_RANKED_OUT`:** Gold evidence exists in the candidate pool, but retriever ranked it below the top-$k$ cutoff ($k=3$).
3. **`F3_ASSEMBLY_LOSS`:** Gold evidence was retrieved in top-$k$, but dropped/truncated before reaching the generator context.
4. **`F4_CONTRADICTION`:** Retrieved context contained mutually contradictory claims that confused the generator.
5. **`REASONING_ERROR`:** All necessary gold evidence was successfully retrieved and present in `generator_context`, but the generator failed to perform multi-hop synthesis, comparison, or extraction correctly.

### Dimension 7: Multiple Plausible Causes (`multiple_plausible_causes`)
When a failure has multiple contributing causes, record secondary factors. Examples:
- `["PARTIAL_RETRIEVAL", "GENERATOR_EXTRACTION_FAILURE"]`: The retriever provided 1 of 2 passages, and the generator failed to synthesize a partial answer from the available passage.
- `["DISTRACTOR_AMBIGUITY", "REASONING_ERROR"]`: Distractor passages contained highly similar named entities that distracted the extractive generator.

### Dimension 8: Unknown / Insufficient Evidence (`is_unknown`)
- Set to `True` if the observational trace lacks sufficient information to substantiate a definitive cause (e.g. gold evidence annotations are ambiguous or passage text is truncated in raw source data).

---

## 4. Reviewer Protocol: Dual-Review & Intra-Rater Reliability

To ensure scientific rigor and avoid idiosyncratic reviewer bias:

### A. Two-Reviewer Protocol (Inter-Annotator Reliability)
1. **Independent Review:** Reviewer 1 and Reviewer 2 independently label a designated subset of natural failure trials ($N \ge 20$). Neither reviewer has access to the other's ratings or to diagnostic predictions.
2. **Agreement Metrics:** Compute:
   - **Percent Agreement ($P_o$):** Proportion of cases where both reviewers assign the identical primary cause:
     $$P_o = \frac{\sum_{i=1}^N \mathbb{I}(\text{cause}_1^{(i)} = \text{cause}_2^{(i)})}{N}$$
   - **Cohen's Kappa ($\kappa$):** Chance-adjusted inter-annotator agreement:
     $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
     where $P_e$ is the expected agreement by chance based on marginal rating frequencies.
3. **Adjudication Procedure:** For any trial where $\text{cause}_1 \ne \text{cause}_2$, an Adjudicator conducts a joint review of both reviewers' written rationales to determine the final consensus label. The discrepancy and final resolution must be logged in `disagreement_summary.json`.

### B. Single-Reviewer Protocol (Intra-Rater Consistency Check)
If only a single human reviewer is available:
1. **Delayed Second Pass:** The reviewer must conduct a second labeling pass after a mandatory delay of **at least 48 hours** with randomized trial presentation order.
2. **Methodological Classification:** The resulting consistency metric is strictly an **intra-rater consistency check** measuring temporal stability. It **must never be reported as inter-annotator agreement** or Cohen's $\kappa$ across independent individuals.

---

## 5. Reviewer Decision Hierarchy Flowchart

```text
                                [Examine Natural Failure]
                                            |
                         Was all gold evidence retrieved in top-k?
                                   /                 \
                                 YES                  NO
                                 /                     \
                   Did evidence survive context?     Was evidence in candidate pool?
                      /                 \                 /                   \
                    YES                  NO             YES                    NO
                    /                     \             /                       \
        Did context contain        [F3_ASSEMBLY_LOSS] [F2_RANKED_OUT]   [F1_UNAVAILABLE]
          contradictions?
              /       \
            YES        NO
            /           \
    [F4_CONTRADICTION] [REASONING_ERROR]
```

---

## 6. Scope Limitations & Threats to Validity

All findings from the natural failure audit must be bounded by the following documented experimental constraints:

1. **Pipeline Architecture Bounds:** The pipeline uses pure BM25 retrieval ($k=3$) and a deterministic extractive mock generator. It does not reflect fine-tuned multi-hop cross-encoders or 70B+ instruction-tuned LLMs.
2. **Task Specificity:** HotpotQA requires 2-hop bridge and comparison reasoning. In multi-hop setups, natural failures are heavily dominated by reasoning errors ($90\%$) where partial evidence is present, whereas single-hop extractive QA benchmarks would exhibit higher retrieval-dominated natural failures.
3. **Diagnostic Transfer Boundary:** Because lightweight diagnostics ($D_1, D_2$) are designed for transit faults (empty lists, score cliffs, truncation diffs), their transfer agreement on natural reasoning failures is expected to be low. This is an informative scientific finding bounding diagnostic utility, not an experimental failure.
