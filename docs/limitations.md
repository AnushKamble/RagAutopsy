# RAG-Autopsy Limitations & Threats to Validity

Every experimental claim in RAG-Autopsy is strictly bounded by the following documented limitations:

---

## 1. Causal Identifiability Limits (F1 vs F2)
- **Top-$k$ Observability Horizon:** Under a strictly blind top-$k$ trace (Regime B), an operational diagnoser observing only the final retrieved passages cannot distinguish between a complete corpus omission (F1) and an evidence rank shortfall (F2). Both produce identical signatures in top-$k$.
- **Mitigation & Finding:** We explicitly formulate this as an empirical finding: in Regime B, diagnostics must abstain (`UNKNOWN: AMBIGUOUS_RETRIEVAL_MISS`) rather than pretending to classify unobservable causes. Distinguishing them requires two-stage candidate pool logging (Regime A).

---

## 2. Generator Reasoning Errors vs Pipeline Transit Faults
- **Transfer Gap on Natural Failures:** Controlled fault injection models pipeline transit interventions (removal, demotion, truncation, contradiction). In our natural failure audit of 100 unmodified questions, **90% of failures were adjudicated as generator reasoning errors** (where adequate evidence was present in context, but the answer failed due to multi-hop synthesis limitations).
- **Finding:** Inexpensive retrieval/lexical diagnostics cannot attribute generator reasoning errors without deeper semantic entailment or generation judges. Synthetic attribution performance cannot be claimed to universally transfer to reasoning-dominated natural failures.

---

## 3. Extractive Baseline vs Large Generative LLMs
- **Core Scope:** To maintain strict CPU reproducibility on any student laptop without GPU or paid API dependencies, the primary benchmark uses BM25 and cached deterministic extractive generators.
- **Threat to Validity:** Modern 70B+ LLMs exhibit stronger parametric memory and instruction following, which may reduce shallow extractive errors but introduce subtle hallucination modes not fully captured by lexical overlap metrics (D2).

---

## 4. Dataset and Domain Fit
- **HotpotQA Distractor Scope:** HotpotQA questions specifically require 2-hop bridge or comparison reasoning across Wikipedia paragraphs. While ideal for evaluating distractor demotion and multi-document assembly, performance on domain-specific corpora (e.g. legal, financial, or biomedical documents with long-range tables) remains an open research question.
