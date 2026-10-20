"""RAG-Autopsy: Interactive Failure Diagnosis Studio (Streamlit Web Dashboard).

Usage:
    streamlit run app.py
"""

from pathlib import Path
import sys
import streamlit as st

# Ensure src is on path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rag_autopsy.core.schemas import DiagnosticTrace, EvaluatorRecord
from rag_autopsy.data.adapter import load_qa_records, load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.diagnostics.baselines import MajorityCauseBaseline, RandomBaseline
from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic
from rag_autopsy.diagnostics.d2_support import D2LexicalSupportDiagnostic
from rag_autopsy.diagnostics.d3_nli import D3NLISupportDiagnostic
from rag_autopsy.diagnostics.d4_policy import D4CombinedAttributionPolicy
from rag_autopsy.faults.f1_unavailable import F1EvidenceUnavailableInjector
from rag_autopsy.faults.f2_ranked_out import F2EvidenceRankedOutInjector
from rag_autopsy.faults.f3_assembly_loss import F3ContextAssemblyLossInjector
from rag_autopsy.faults.f4_contradiction import F4ContradictoryEvidenceInjector
from rag_autopsy.faults.f5_unsupported import F5UnsupportedGenerationInjector
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import MockExtractiveGenerator
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever


REPO_ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_data():
    p = REPO_ROOT / ".data" / "processed" / "hotpot_subset_100.json"
    if p.is_file():
        try:
            return load_qa_records(p)
        except Exception:
            pass

    synth = REPO_ROOT / "tests" / "fixtures" / "synthetic_qa.json"
    if synth.is_file():
        try:
            return parse_hotpot_records(load_raw_hotpot_json(synth))
        except Exception:
            pass

    # Safe fallback if neither file is found
    from rag_autopsy.data.schema import PassageRecord, QARecord

    return [
        QARecord(
            question_id="demo_fallback_01",
            question="Where is Bubblegum Alley located in California?",
            gold_answer="San Luis Obispo, California",
            gold_evidence_passage_ids=["p1"],
            passages=[
                PassageRecord(
                    passage_id="p1",
                    title="Bubblegum Alley",
                    text="Bubblegum Alley is a tourist attraction in downtown San Luis Obispo, California.",
                    sentences=["Bubblegum Alley is a tourist attraction in downtown San Luis Obispo, California."],
                ),
                PassageRecord(
                    passage_id="p2",
                    title="Gum Wall",
                    text="The Market Theater Gum Wall is a brick wall covered in used chewing gum in Seattle.",
                    sentences=["The Market Theater Gum Wall is a brick wall covered in used chewing gum in Seattle."],
                ),
            ],
        )
    ]


def main():
    st.set_page_config(page_title="RAG-Autopsy Studio", page_icon="🔬", layout="wide")
    st.title("🔬 RAG-Autopsy: Interactive Failure Diagnosis Studio")
    st.markdown(
        "**Fault-Injection-Validated Failure Attribution for Retrieval-Augmented Generation.** "
        "Strictly decoupled into unprivileged operational traces and privileged evaluator records."
    )

    records = load_data()
    q_map = {f"[{r.question_id[:8]}] {r.question[:80]}...": r for r in records}

    # Sidebar Controls
    st.sidebar.header("🕹️ Simulation Settings")
    selected_label = st.sidebar.selectbox("Select Test Query:", list(q_map.keys()))
    rec = q_map[selected_label]

    regime = st.sidebar.radio(
        "Observability Regime:",
        ["regime_a", "regime_b"],
        format_func=lambda x: "Regime A (Candidate Pool Visible)" if x == "regime_a" else "Regime B (Blind Top-k)",
    )

    fault = st.sidebar.selectbox(
        "Injected Fault Intervention:",
        ["F0 (Clean Control)", "F1 (Evidence Unavailable)", "F2 (Evidence Ranked Out)", "F3 (Assembly Loss)", "F4 (Conflicting Evidence)", "F5 (Unsupported Generation)"],
    )
    fault_code = fault.split()[0]

    top_k = st.sidebar.slider("Top-k Retrieval Cutoff:", min_value=1, max_value=5, value=3)

    # Simulation execution
    base_corpus = Corpus.from_qa_records([rec])
    builder = ContextBuilder(max_context_chars=3000)
    gen = MockExtractiveGenerator()

    # Apply Intervention
    trial_corpus = base_corpus
    pre_passages = None
    injected_applied = False
    injected_valid = True
    inv_reason = None
    override_ans = None

    if fault_code == "F1":
        f1_inj = F1EvidenceUnavailableInjector()
        res_f1 = f1_inj.inject(rec, base_corpus)
        trial_corpus = res_f1.modified_corpus or base_corpus
        injected_applied = res_f1.injection_applied
        injected_valid = res_f1.trial_valid
        inv_reason = res_f1.invalid_reason

    elif fault_code == "F2":
        f2_inj = F2EvidenceRankedOutInjector(distractor_count=5, cutoff_k=top_k)
        res_f2 = f2_inj.inject(rec, base_corpus)
        trial_corpus = res_f2.modified_corpus or base_corpus
        injected_applied = res_f2.injection_applied
        injected_valid = res_f2.trial_valid
        inv_reason = res_f2.invalid_reason

    retriever = BM25Retriever(trial_corpus)
    retrieved, candidate_scores = retriever.retrieve(rec.question, top_k=top_k, candidate_pool_size=20)

    if regime == "regime_b":
        candidate_scores = None

    if fault_code == "F3":
        f3_inj = F3ContextAssemblyLossInjector()
        surviving, res_f3 = f3_inj.apply_to_retrieved_passages(retrieved, rec.gold_evidence_passage_ids)
        ctx, pre_ids, post_ids = builder.build_context(surviving)
        injected_applied = res_f3.injection_applied
        injected_valid = res_f3.trial_valid
        inv_reason = res_f3.invalid_reason
    elif fault_code == "F4":
        f4_inj = F4ContradictoryEvidenceInjector()
        modified_f4, res_f4 = f4_inj.apply_to_context_passages(rec, retrieved)
        ctx, pre_ids, post_ids = builder.build_context(modified_f4)
        retrieved = modified_f4
        injected_applied = res_f4.injection_applied
        injected_valid = res_f4.trial_valid
        inv_reason = res_f4.invalid_reason
    else:
        ctx, pre_ids, post_ids = builder.build_context(retrieved)

    if fault_code == "F5":
        f5_inj = F5UnsupportedGenerationInjector()
        override_ans, res_f5 = f5_inj.apply_to_generation(rec, retrieved, ctx)
        injected_applied = res_f5.injection_applied
        injected_valid = res_f5.trial_valid
        inv_reason = res_f5.invalid_reason

    ans = override_ans or gen.generate(rec.question, ctx)

    trace = DiagnosticTrace(
        trial_id="studio_interactive_trial",
        question_id=rec.question_id,
        question=rec.question,
        retrieved_top_k=retrieved,
        candidate_pool_scores=candidate_scores,
        source_passage_ids=[p.passage_id for p in retrieved],
        pre_assembly_passage_ids=pre_ids,
        post_assembly_passage_ids=post_ids,
        generator_context=ctx,
        answer=ans,
        latency_ms=12.5,
        generator_model="MockExtractiveGenerator (Testing Infrastructure)",
    )

    eval_rec = EvaluatorRecord(
        trial_id="studio_interactive_trial",
        question_id=rec.question_id,
        split="interactive",
        injected_fault=fault_code,
        gold_answers=[rec.gold_answer],
        gold_evidence_ids=rec.gold_evidence_passage_ids,
        injection_applied=injected_applied,
        trace_condition_achieved=True,
        trial_valid=injected_valid,
        invalid_reason=inv_reason,
    )

    # Layout: Pipeline Observables vs Diagnostic Predictions
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📡 Unprivileged Diagnostic Trace (Observable)")
        st.info(f"**Query:** {rec.question}")
        st.write(f"**Answer Output:** `{trace.answer}`")

        with st.expander("🔍 Top-k Retrieved Passages", expanded=True):
            for p in trace.retrieved_top_k:
                st.markdown(f"**Rank {p.rank}** (`{p.passage_id}`) — Score: `{p.score}`")
                st.caption(p.text[:300] + "...")

        with st.expander("📦 Context Assembly Stream"):
            st.write(f"**Pre-Assembly Passage IDs:** `{trace.pre_assembly_passage_ids}`")
            st.write(f"**Post-Assembly Passage IDs:** `{trace.post_assembly_passage_ids}`")
            if len(trace.pre_assembly_passage_ids) > len(trace.post_assembly_passage_ids):
                st.error("⚠️ Dropped Passages Detected in Transit!")
            else:
                st.success("✅ Context Assembled Intact")

        if trace.candidate_pool_scores:
            with st.expander("📊 Candidate Pool Score Distribution (Regime A)"):
                st.bar_chart(trace.candidate_pool_scores)
        else:
            st.warning("🙈 Regime B: Candidate Pool Scores Hidden from Diagnoser")

    with col2:
        st.subheader("🩺 Diagnostic Attribution Engine")

        d1 = D1RetrievalDiagnostic(regime=regime)
        d2 = D2LexicalSupportDiagnostic()
        d3 = D3NLISupportDiagnostic()
        d4 = D4CombinedAttributionPolicy(regime=regime, enable_nli=True)

        diag_d1 = d1.diagnose(trace)
        diag_d2 = d2.diagnose(trace)
        diag_d3 = d3.diagnose(trace)
        diag_d4 = d4.diagnose(trace)

        st.markdown("#### Diagnostic Attribution Comparison")
        st.table([
            {"Diagnostic Policy": "D1 Retrieval Rules", "Predicted Cause": diag_d1.predicted_cause, "Confidence": f"{diag_d1.confidence:.2f}"},
            {"Diagnostic Policy": "D2 Lexical Support", "Predicted Cause": diag_d2.predicted_cause, "Confidence": f"{diag_d2.confidence:.2f}"},
            {"Diagnostic Policy": "D3 NLI Model", "Predicted Cause": diag_d3.predicted_cause, "Confidence": f"{diag_d3.confidence:.2f}"},
            {"Diagnostic Policy": "D4 Combined Policy", "Predicted Cause": diag_d4.predicted_cause, "Confidence": f"{diag_d4.confidence:.2f}"},
        ])

        st.markdown(f"**D4 Attribution Rationale:** _{diag_d4.evidence_rationale}_")

        st.subheader("🔐 Evaluator Reveal (Privileged Ground Truth)")
        is_match = (diag_d4.predicted_cause == eval_rec.injected_fault)
        st.write(f"**True Injected Fault:** `{eval_rec.injected_fault}`")
        st.write(f"**Gold Answer:** `{rec.gold_answer}`")
        st.write(f"**Adjudicated Valid:** `{eval_rec.trial_valid}`")

        if is_match:
            st.success("🎯 Attribution Match: TRUE (Diagnoser recovered true fault without leakage)")
        else:
            if diag_d4.abstained:
                st.warning("⚠️ Diagnoser Abstained (Ambiguous signature)")
            else:
                st.error("❌ Attribution Mismatch")


if __name__ == "__main__":
    main()
