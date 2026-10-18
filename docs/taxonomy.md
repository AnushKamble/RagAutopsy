# RAG-Autopsy Failure Taxonomy & Identifiability Matrix

This document defines the formal failure taxonomy, controlled injection mechanisms, observable trace signatures, and identifiability conditions across observability regimes.

---

## 1. Primary Intervention Taxonomy

An injected label represents the **controlled intervention applied**, not necessarily the sole explanation for downstream generation.

| ID | Intervention | Injection Mechanism | Observable Trace Signature | Identifiability / Ambiguity |
|---|---|---|---|---|
| **F0** | **Clean Control** | No intentional intervention. Baseline execution on searchable corpus. | Normal retrieval scores; evidence passages present in generator context; high lexical/NLI support. | Baseline control. May still fail naturally due to generator reasoning limitations. |
| **F1** | **Evidence Unavailable** | True supporting passages are removed from the searchable candidate pool. | Depressed candidate pool scores (Regime A); absence of gold terms in retrieved context; low context support. | **Ambiguous with F2 under blind top-$k$ (Regime B).** Resolved in Regime A via candidate pool score cliff probe. |
| **F2** | **Evidence Ranked Out** | High-scoring query-targeted distractors injected into index to push gold evidence below cutoff $k$. | Evidence remains in corpus/candidate pool; top-$k$ occupied by distractors; continuous score decay into ranks $> k$. | Under Regime B, looks identical to F1 in top-$k$. Under Regime A, distinguishable via candidate pool score distribution. |
| **F3** | **Context Assembly Loss** | Evidence retrieved in top-$k$, but truncated or dropped during prompt context assembly. | Structural difference: `len(pre_assembly) > len(post_assembly)`; passage ID present in retrieved set but missing in generator context. | Fully observable via pre/post assembly passage ID trace. |
| **F4** | **Contradictory Evidence** | Explicitly contradictory passage injected into context alongside true evidence. | Contradiction cues detected in context; conflicting claims across retrieved passages; answer vacillates or repeats conflict. | Observable through lexical contradiction cues or D3 NLI cross-encoder scores. |
| **F5** | **Unsupported Generation** | Evidence is present and intact in generator context, but generator hallucinates ungrounded answer. | Normal retrieval; evidence intact in context; answer has low lexical overlap and unentailed NLI score. | Distinguishable from F1/F3 because generator context actually contains the required evidence. |

---

## 2. Compound / Multi-Stage Faults

Real-world production failures often span multiple pipeline stages. RAG-Autopsy models these through sequential composition:

- **F2+F3 (Ranked Out + Assembly Loss):** Distractors demote evidence down the rank order, and assembly length limits truncate remaining evidence from the context window.
- **F3+F4 (Assembly Loss + Contradiction):** True evidence is dropped from the context window, leaving an injected contradictory claim as the sole context seen by the generator.

---

## 3. Identifiability Matrix (Regime A vs Regime B)

The table below details whether every pair of failure modes can be distinguished from visible trace fields alone, without privileged evaluator metadata:

| Fault Pair | Regime A (Candidate Pool Trace) | Regime B (Blind Top-$k$ Trace) | Distinguishing Visible Trace Features |
|---|---|---|---|
| **F0 vs F1** | Observable | Observable | Lexical context support, answer-to-context overlap, candidate pool score drop. |
| **F0 vs F2** | Observable | Observable | Context support, presence of distractors in top-$k$. |
| **F1 vs F2** | **Observable** | **Fundamentally Unidentifiable** | In Regime A, candidate pool max score and score cliff distinguish absent vs demoted evidence. In Regime B, diagnoser merges into `RETRIEVAL_MISS` or abstains. |
| **F2 vs F3** | Observable | Observable | In F2, evidence never reaches top-$k$. In F3, evidence is in top-$k$ but dropped in `post_assembly_passage_ids`. |
| **F3 vs F4** | Observable | Observable | F3 has pre/post assembly ID length mismatch; F4 has contradiction passage in context. |
| **F3 vs F5** | Observable | Observable | In F3, context lacks evidence (dropped). In F5, context contains evidence, but answer is ungrounded. |
| **F4 vs F5** | Observable | Observable | F4 has internal context conflict; F5 context is internally consistent but answer diverges. |
