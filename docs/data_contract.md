# RAG-Autopsy Data Contract

## 1. Primary Dataset Specification

- **Dataset Name:** HotpotQA (Distractor Setting, Development Split `v1.1`)
- **Source Citation:** Yang et al., 2018. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP 2018. [https://hotpotqa.github.io/](https://hotpotqa.github.io/)
- **License / Terms:** Creative Commons Attribution-ShareAlike 4.0 International ([CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)). Permitted for academic and commercial research with attribution.
- **Data File Format:** JSON array of dictionaries.
- **Download Footprint:** Development set (`hotpot_dev_distractor_v1.json`) is ~45 MB uncompressed.

---

## 2. Storage and Git Hygiene Policy

1. **Raw Data Isolation:**
   - Raw downloaded files must reside exclusively in `.data/raw/` (ignored by Git).
   - Machine-processed subsets and split manifests reside in `.data/processed/` and `configs/`.
   - Never commit raw external datasets to Git.
2. **Deterministic CI Fixture:**
   - A synthetic fixture of $N=2$ questions, 20 paragraphs conforming to the HotpotQA schema is stored at `tests/fixtures/synthetic_qa.json` for offline automated tests.

---

## 3. Schema & Column Specifications

Each raw record in the dataset conforms to the following schema:

| Field | Type | Description |
|---|---|---|
| `_id` | `str` | Unique 24-character hex identifier (e.g. `5a8b57f25542995d1e6f1371`). Primary key. |
| `question` | `str` | Natural language question requiring multi-hop reasoning. |
| `answer` | `str` | Gold reference answer string. |
| `type` | `str` | Question type: `"bridge"` or `"comparison"`. |
| `level` | `str` | Difficulty level: `"easy"`, `"medium"`, or `"hard"`. |
| `supporting_facts` | `List[Tuple[str, int]]` | Pairs of `[title, sentence_index]` indicating sentences required to deduce the answer. |
| `context` | `List[Tuple[str, List[str]]]` | Exactly 10 paragraphs formatted as `[title, [sentence0, sentence1, ...]]`. Exactly 2 paragraphs contain supporting facts; 8 are hard distractors. |

---

## 4. Evidence Set Definition & Normalization

For the purpose of RAG-Autopsy pipeline benchmarking:
- **Corpus Unit (Passage):** Each title and its list of sentences is combined into a single discrete passage.
  - `passage_id`: Formatted deterministically as `{item_id}__{slug(title)}`.
  - `text`: Formatted as `"{title}\n" + " ".join(sentences)`.
- **Gold Evidence Passages:** The subset of passages in `context` whose titles appear in `supporting_facts`. In the HotpotQA distractor setting, there are exactly 2 gold evidence passages per query.
- **Passage-Level Grounding:** Retrieval is considered to have retrieved evidence if at least one gold passage is retrieved in top-$k$. Full retrieval occurs when both gold passages are present in top-$k$.

---

## 5. Grouped Partitioning and Split Integrity

To prevent cross-partition contamination when fault variants are generated:
- **Grouping Unit:** Split partitioning is performed strictly at the **`_id` (Question ID)** level.
- **Ratios:** 60% Development (`dev`), 20% Calibration (`calibration`), 20% Held-Out Test (`test`).
- **Integrity Invariant:**
  $$\text{IDs}(\text{dev}) \cap \text{IDs}(\text{calibration}) = \emptyset$$
  $$\text{IDs}(\text{dev}) \cap \text{IDs}(\text{test}) = \emptyset$$
  $$\text{IDs}(\text{calibration}) \cap \text{IDs}(\text{test}) = \emptyset$$
- All fault-injected trials derived from a source question (F0, F1, F2, F3, F4) inherit the partition of their parent question.
- Calibration thresholds (e.g. for D1 score gaps or D2 overlap cutoffs) are fit *only* on the calibration partition and frozen before test evaluation.
