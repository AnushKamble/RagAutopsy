# Architecture and Experimental Decision Log

This log records all consequential architecture, research-design, and methodological decisions for RAG-Autopsy.

---

## ADR-001: Resolution of F1 vs F2 Causal Identifiability
- **Date:** 2026-10-18
- **Status:** Approved
- **Context:** In standard operational RAG logging, if only the final top-$k$ retrieved passages are recorded, F1 (evidence absent from the index) and F2 (evidence present in the index but ranked below cutoff $k$) present identical observable traces to a diagnostic: top-$k$ contains only distractors, context lacks evidence, and generation fails. Forcing a classifier to distinguish them from top-$k$ observations alone introduces shortcut learning or requires leaking privileged corpus membership flags.
- **Decision:** We pre-register two distinct evaluation regimes:
  1. **Regime A (Candidate Pool Trace):** Modern two-stage RAG systems realistically log broad candidate pool scores (e.g. top $K_{\text{pool}}=20$). Diagnoser inspects the score distribution across top-20 candidates without knowing gold passage IDs. In F1, max candidate score is severely depressed (score cliff). In F2, evidence exists in the candidate pool, exhibiting a continuous decay curve into ranks $k+1 \dots K$.
  2. **Regime B (Strict Top-$k$ Blind Diagnoser):** Under top-$k$-only observability, the diagnostic taxonomy explicitly merges F1 and F2 into a single identifiable category: `RETRIEVAL_MISS` (or abstains with `UNKNOWN: AMBIGUOUS_RETRIEVAL_MISS`).
- **Consequences:** Eliminates leakage of gold evidence IDs or hidden corpus status. Preserves experimental validity.

---

## ADR-002: Dataset Selection — HotpotQA Distractor (Dev) with SQuAD Fallback
- **Date:** 2026-10-18
- **Status:** Approved
- **Context:** Need an evidence-annotated QA dataset suitable for laptop execution, with clean license terms, unambiguous evidence IDs, and distractor paragraphs for F1/F2 injection.
- **Decision:** Select HotpotQA Development Split (distractor setting, `v1.1`, CC BY-SA 4.0).
  - Explicit sentence- and paragraph-level `supporting_facts`.
  - Built-in 10-paragraph candidate pool per query (2 gold, 8 distractors).
  - Dev split is ~45 MB JSON, easily processed in memory.
  - SQuAD 1.1 pooled articles documented as fallback.
- **Consequences:** We must define passage-level evidence sets $\{p_{\text{gold}_1}, p_{\text{gold}_2}\}$ to cleanly accommodate multi-hop queries.

---

## ADR-003: CPU-First Core Stack and Dependency Policy
- **Date:** 2026-10-18
- **Status:** Approved
- **Context:** Need a setup that runs fast and deterministically on a standard student laptop without requiring paid APIs or GPU access. System currently has 16GB RAM with ~2.1GB free.
- **Decision:**
  - Package manager: `uv` with pinned Python 3.11 dependencies.
  - Core retriever: pure Python/NumPy BM25 via `rank-bm25`.
  - Generator: pluggable interface with hash-keyed disk caching (`sha256(prompt + context + config)`). Tests use a deterministic extractive mock generator.
  - Diagnostics: D1 (retrieval score/rank rules) and D2 (lexical/entity overlap) as core. D3 (NLI) is an optional extra (`[nli]`).
- **Consequences:** Zero external API costs, zero GPU memory pressure, sub-second unit test execution.
