# Data Directory & Provenance

This directory documents the data sources, licenses, and preprocessing rules for the RAG-Autopsy benchmark.

---

## 1. Dataset Provenance & Terms

- **Source Dataset:** [HotpotQA](https://hotpotqa.github.io/) (Development split, distractor setting, `v1.1`).
- **Citation:** Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018.
- **License:** Creative Commons Attribution-ShareAlike 4.0 International ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)).
- **Intended Use:** Multi-hop question answering evaluation, retrieval ranking benchmark, and controlled fault injection.

---

## 2. Directory Structure

```text
.data/
  raw/
    hotpot_dev_distractor_v1.json    # Upstream HotpotQA development distractor split
  processed/
    hotpot_subset_100.json           # Deterministic 100-question sample with validated schema
configs/
  split_manifest_100.json            # Deterministic, pairwise-disjoint question split manifest
```

---

## 3. Data Invariants

1. **Strict Question-Level Grouping:** Every fault variant of a question ($F0 \dots F5$) resides in the exact same partition (Dev, Calibration, or Test). Cross-split question leakage is strictly forbidden and verified by automated unit tests (`tests/test_splits.py`).
2. **Stable Passage Identifiers:** Passages are assigned stable, non-leaking identifiers formatted as `{question_id}__p_{index}_{slug}`. Passage identifiers never encode fault labels or ground-truth evidence status.
3. **Immutability:** Injectors never mutate raw or processed files on disk. All interventions produce trial-scoped copy-on-write memory corpora.

---

## 4. Supplementary Datasets

- **RAGTruth:** ParticleMedia/RAGTruth (MIT License / CC). Used strictly for benchmarking hallucination/grounding diagnostics (`tests/fixtures/ragtruth_synthetic.json`), not for pipeline stage attribution.
