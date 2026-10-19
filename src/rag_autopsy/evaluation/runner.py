import random
import time
from typing import Any, Dict, List, Optional, Tuple
from rag_autopsy.core.schemas import (
    DiagnosticTrace,
    EvaluatedTrial,
    EvaluatorRecord,
    EvaluatorStore,
    explicit_evaluation_join,
    generate_stable_trial_id,
)
from rag_autopsy.data.schema import QARecord

from rag_autopsy.data.splits import SplitManifest
from rag_autopsy.diagnostics.baselines import MajorityCauseBaseline, RandomBaseline
from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic
from rag_autopsy.diagnostics.d2_support import D2LexicalSupportDiagnostic
from rag_autopsy.diagnostics.d4_policy import D4CombinedAttributionPolicy
from rag_autopsy.evaluation.attribution_metrics import compute_attribution_metrics
from rag_autopsy.evaluation.bootstrap import clustered_bootstrap_attribution
from rag_autopsy.evaluation.validity import adjudicate_trials
from rag_autopsy.faults.f1_unavailable import F1EvidenceUnavailableInjector
from rag_autopsy.faults.f2_ranked_out import F2EvidenceRankedOutInjector
from rag_autopsy.faults.f3_assembly_loss import F3ContextAssemblyLossInjector
from rag_autopsy.faults.f4_contradiction import F4ContradictoryEvidenceInjector
from rag_autopsy.faults.f5_unsupported import F5UnsupportedGenerationInjector
from rag_autopsy.diagnostics.d3_nli import D3NLISupportDiagnostic
from rag_autopsy.diagnostics.d4_learned import D4LearnedClassifierPolicy
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import BaseGenerator, MockExtractiveGenerator
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever


def _generate_split_trials(
    records: List[QARecord],
    split_name: str,
    top_k: int = 3,
    candidate_pool_size: int = 20,
    seed: int = 42,
    generator: Optional[BaseGenerator] = None,
    include_f5: bool = False,
) -> Tuple[List[DiagnosticTrace], List[EvaluatorRecord]]:
    """Generate diagnostic traces and privileged evaluator records across a partition of QA records."""
    base_corpus = Corpus.from_qa_records(records)
    gen = generator or MockExtractiveGenerator()
    builder = ContextBuilder(max_context_chars=3000)

    f1_injector = F1EvidenceUnavailableInjector()
    f2_injector = F2EvidenceRankedOutInjector(distractor_count=5, cutoff_k=top_k)
    f3_injector = F3ContextAssemblyLossInjector()
    f4_injector = F4ContradictoryEvidenceInjector()
    f5_injector = F5UnsupportedGenerationInjector()

    traces: List[DiagnosticTrace] = []
    eval_records: List[EvaluatorRecord] = []

    for rec in records:
        # F0: Clean control
        id_0 = generate_stable_trial_id(seed=seed, question_id=rec.question_id, intervention_key="F0")
        pipe_f0 = RAGPipeline(
            retriever=BM25Retriever(base_corpus),
            context_builder=builder,
            generator=gen,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
        tr_0, ev_0 = pipe_f0.run(
            trial_id=id_0,
            question_id=rec.question_id,
            question=rec.question,
            split=split_name,
            gold_answers=[rec.gold_answer],
            gold_evidence_ids=rec.gold_evidence_passage_ids,
            injected_fault="F0",
            injection_applied=False,
            trace_condition_achieved=True,
            trial_valid=True,
        )
        traces.append(tr_0)
        eval_records.append(ev_0)

        # F1: Evidence Unavailable
        id_1 = generate_stable_trial_id(seed=seed, question_id=rec.question_id, intervention_key="F1")
        res_f1 = f1_injector.inject(rec, base_corpus)
        trial_corpus_f1 = res_f1.modified_corpus or base_corpus  # type: ignore[attr-defined]
        pipe_f1 = RAGPipeline(
            retriever=BM25Retriever(trial_corpus_f1),
            context_builder=builder,
            generator=gen,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
        tr_1, ev_1 = pipe_f1.run(
            trial_id=id_1,
            question_id=rec.question_id,
            question=rec.question,
            split=split_name,
            gold_answers=[rec.gold_answer],
            gold_evidence_ids=rec.gold_evidence_passage_ids,
            injected_fault="F1",
            fault_parameters=res_f1.fault_parameters,
            injection_applied=res_f1.injection_applied,
            trace_condition_achieved=res_f1.trace_condition_achieved,
            trial_valid=res_f1.trial_valid,
            invalid_reason=res_f1.invalid_reason,
        )
        traces.append(tr_1)
        eval_records.append(ev_1)

        # F2: Evidence Ranked Out
        id_2 = generate_stable_trial_id(seed=seed, question_id=rec.question_id, intervention_key="F2")
        res_f2 = f2_injector.inject(rec, base_corpus)
        trial_corpus_f2 = res_f2.modified_corpus or base_corpus  # type: ignore[attr-defined]
        pipe_f2 = RAGPipeline(
            retriever=BM25Retriever(trial_corpus_f2),
            context_builder=builder,
            generator=gen,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
        tr_2, ev_2 = pipe_f2.run(
            trial_id=id_2,
            question_id=rec.question_id,
            question=rec.question,
            split=split_name,
            gold_answers=[rec.gold_answer],
            gold_evidence_ids=rec.gold_evidence_passage_ids,
            injected_fault="F2",
            fault_parameters=res_f2.fault_parameters,
            injection_applied=res_f2.injection_applied,
            trace_condition_achieved=res_f2.trace_condition_achieved,
            trial_valid=res_f2.trial_valid,
            invalid_reason=res_f2.invalid_reason,
        )
        traces.append(tr_2)
        eval_records.append(ev_2)

        # F3: Context Assembly Loss
        id_3 = generate_stable_trial_id(seed=seed, question_id=rec.question_id, intervention_key="F3")
        pipe_f3_base = RAGPipeline(
            retriever=BM25Retriever(base_corpus),
            context_builder=builder,
            generator=gen,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
        retrieved_f3, pool_scores_f3 = pipe_f3_base.retriever.retrieve(
            query=rec.question, top_k=top_k, candidate_pool_size=candidate_pool_size
        )
        surviving_f3, res_f3 = f3_injector.apply_to_retrieved_passages(
            retrieved_f3, rec.gold_evidence_passage_ids
        )
        ctx_f3, _, post_f3 = builder.build_context(surviving_f3)
        ans_f3 = gen.generate(rec.question, ctx_f3)
        tr_3 = DiagnosticTrace(
            trial_id=id_3,
            question_id=rec.question_id,
            question=rec.question,
            retrieved_top_k=retrieved_f3,
            candidate_pool_scores=pool_scores_f3,
            source_passage_ids=[p.passage_id for p in retrieved_f3],
            pre_assembly_passage_ids=[p.passage_id for p in retrieved_f3],
            post_assembly_passage_ids=post_f3,
            generator_context=ctx_f3,
            answer=ans_f3,
            latency_ms=10.0,
            generator_model=gen.get_model_name(),
            generator_config=gen.get_config(),
        )
        ev_3 = EvaluatorRecord(
            trial_id=id_3,
            question_id=rec.question_id,
            split=split_name,
            injected_fault="F3",
            fault_parameters=res_f3.fault_parameters,
            gold_answers=[rec.gold_answer],
            gold_evidence_ids=rec.gold_evidence_passage_ids,
            injection_applied=res_f3.injection_applied,
            trace_condition_achieved=res_f3.trace_condition_achieved,
            trial_valid=res_f3.trial_valid,
            invalid_reason=res_f3.invalid_reason,
        )
        traces.append(tr_3)
        eval_records.append(ev_3)

        # F4: Contradictory Evidence
        id_4 = generate_stable_trial_id(seed=seed, question_id=rec.question_id, intervention_key="F4")
        pipe_f4_base = RAGPipeline(
            retriever=BM25Retriever(base_corpus),
            context_builder=builder,
            generator=gen,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
        retrieved_f4, pool_scores_f4 = pipe_f4_base.retriever.retrieve(
            query=rec.question, top_k=top_k, candidate_pool_size=candidate_pool_size
        )
        modified_f4, res_f4 = f4_injector.apply_to_context_passages(rec, retrieved_f4)
        ctx_f4, pre_f4, post_f4 = builder.build_context(modified_f4)
        ans_f4 = gen.generate(rec.question, ctx_f4)
        tr_4 = DiagnosticTrace(
            trial_id=id_4,
            question_id=rec.question_id,
            question=rec.question,
            retrieved_top_k=modified_f4,
            candidate_pool_scores=pool_scores_f4,
            source_passage_ids=[p.passage_id for p in modified_f4],
            pre_assembly_passage_ids=pre_f4,
            post_assembly_passage_ids=post_f4,
            generator_context=ctx_f4,
            answer=ans_f4,
            latency_ms=10.0,
            generator_model=gen.get_model_name(),
            generator_config=gen.get_config(),
        )
        ev_4 = EvaluatorRecord(
            trial_id=id_4,
            question_id=rec.question_id,
            split=split_name,
            injected_fault="F4",
            fault_parameters=res_f4.fault_parameters,
            gold_answers=[rec.gold_answer],
            gold_evidence_ids=rec.gold_evidence_passage_ids,
            injection_applied=res_f4.injection_applied,
            trace_condition_achieved=res_f4.trace_condition_achieved,
            trial_valid=res_f4.trial_valid,
            invalid_reason=res_f4.invalid_reason,
        )
        traces.append(tr_4)
        eval_records.append(ev_4)

        # F5: Unsupported Generation / Hallucination (Optional / Extended)
        if include_f5:
            id_5 = generate_stable_trial_id(seed=seed, question_id=rec.question_id, intervention_key="F5")
            pipe_f5_base = RAGPipeline(
                retriever=BM25Retriever(base_corpus),
                context_builder=builder,
                generator=gen,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
            )
            retrieved_f5, pool_scores_f5 = pipe_f5_base.retriever.retrieve(
                query=rec.question, top_k=top_k, candidate_pool_size=candidate_pool_size
            )
            ctx_f5, pre_f5, post_f5 = builder.build_context(retrieved_f5)
            ans_f5, res_f5 = f5_injector.apply_to_generation(rec, retrieved_f5, ctx_f5)
            tr_5 = DiagnosticTrace(
                trial_id=id_5,
                question_id=rec.question_id,
                question=rec.question,
                retrieved_top_k=retrieved_f5,
                candidate_pool_scores=pool_scores_f5,
                source_passage_ids=[p.passage_id for p in retrieved_f5],
                pre_assembly_passage_ids=pre_f5,
                post_assembly_passage_ids=post_f5,
                generator_context=ctx_f5,
                answer=ans_f5,
                latency_ms=10.0,
                generator_model=gen.get_model_name(),
                generator_config=gen.get_config(),
            )
            ev_5 = EvaluatorRecord(
                trial_id=id_5,
                question_id=rec.question_id,
                split=split_name,
                injected_fault="F5",
                fault_parameters=res_f5.fault_parameters,
                gold_answers=[rec.gold_answer],
                gold_evidence_ids=rec.gold_evidence_passage_ids,
                injection_applied=res_f5.injection_applied,
                trace_condition_achieved=res_f5.trace_condition_achieved,
                trial_valid=res_f5.trial_valid,
                invalid_reason=res_f5.invalid_reason,
            )
            traces.append(tr_5)
            eval_records.append(ev_5)

    return traces, eval_records


def run_experiment_harness(
    qa_records: List[QARecord],
    split_manifest: SplitManifest,
    target_split: str = "test",
    regime: str = "regime_a",
    top_k: int = 3,
    candidate_pool_size: int = 20,
    n_bootstraps: int = 100,
    seed: int = 42,
    generator: Optional[BaseGenerator] = None,
    include_f5: bool = False,
    include_d3: bool = False,
    include_d4_learned: bool = False,
    low_score_threshold: float = 2.0,
    score_gap_threshold: float = 1.0,
    min_overlap_ratio: float = 0.3,
) -> Dict[str, Any]:
    """Execute end-to-end fault-injection experiment and diagnostic attribution benchmark."""
    # 1. Filter QA records for target partition
    target_ids = set(
        split_manifest.test_ids
        if target_split == "test"
        else (split_manifest.calibration_ids if target_split == "calibration" else split_manifest.dev_ids)
    )
    records = [r for r in qa_records if r.question_id in target_ids]
    if not records:
        raise ValueError(f"No records found for target split '{target_split}'.")

    # 2. Generate trials for target partition
    traces, eval_records = _generate_split_trials(
        records=records,
        split_name=target_split,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
        generator=generator,
        include_f5=include_f5,
    )

    # 3b. Decouple presentation order so diagnostics cannot exploit sort order or modulo indices
    rng = random.Random(seed)
    paired = list(zip(traces, eval_records))
    rng.shuffle(paired)
    traces = [t for t, _ in paired]
    eval_records = [e for _, e in paired]

    # Store evaluator records in the privileged EvaluatorStore
    evaluator_store = EvaluatorStore()
    for ev in eval_records:
        evaluator_store.add(ev)

    # 4. Adjudicate validity
    valid_evals, invalid_evals, validity_summary = adjudicate_trials(eval_records)
    valid_trial_ids = set(e.trial_id for e in valid_evals)

    # Filter traces for valid trials only
    valid_traces = [t for t in traces if t.trial_id in valid_trial_ids]
    valid_eval_map = {e.trial_id: e for e in valid_evals}

    # 5. Initialize diagnostics under test
    candidate_causes = ["F0", "F1", "F2", "F3", "F4"]
    if include_f5:
        candidate_causes.append("F5")


    # Baseline & policy fitting: When evaluating on held-out test split,
    # fit learned policies and baselines strictly on the training partition (dev/calibration)
    # to guarantee zero train-on-test leakage.
    train_ids = set(split_manifest.dev_ids if split_manifest.dev_ids else split_manifest.calibration_ids)
    train_records = [r for r in qa_records if r.question_id in train_ids] if (target_split == "test" and train_ids) else []

    if train_records:
        train_traces, train_evals = _generate_split_trials(
            records=train_records,
            split_name="dev",
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            seed=seed,
            generator=generator,
            include_f5=include_f5,
        )
        valid_train_evals, _, _ = adjudicate_trials(train_evals)
        valid_train_ids = set(e.trial_id for e in valid_train_evals)
        valid_train_traces = [t for t in train_traces if t.trial_id in valid_train_ids]
        valid_train_eval_map = {e.trial_id: e for e in valid_train_evals}
        train_y = [valid_train_eval_map[t.trial_id].injected_fault for t in valid_train_traces]
    else:
        valid_train_traces = valid_traces
        train_y = [valid_eval_map[t.trial_id].injected_fault for t in valid_traces]

    majority_baseline = MajorityCauseBaseline()
    majority_baseline.fit(train_y)

    random_baseline = RandomBaseline(candidate_causes=candidate_causes, seed=seed)
    d1 = D1RetrievalDiagnostic(
        regime=regime,
        low_score_threshold=low_score_threshold,
        score_gap_threshold=score_gap_threshold,
    )
    d2 = D2LexicalSupportDiagnostic(min_overlap_ratio=min_overlap_ratio)
    d4 = D4CombinedAttributionPolicy(
        regime=regime,
        low_score_threshold=low_score_threshold,
        score_gap_threshold=score_gap_threshold,
        min_overlap_ratio=min_overlap_ratio,
        enable_nli=include_d3,
    )

    diagnostics = [majority_baseline, random_baseline, d1, d2, d4]

    if include_d3:
        d3 = D3NLISupportDiagnostic()
        diagnostics.append(d3)

    if include_d4_learned:
        d4_learned = D4LearnedClassifierPolicy(classifier_type="decision_tree", random_state=seed)
        d4_learned.fit(valid_train_traces, train_y)
        diagnostics.append(d4_learned)

    # 6. Run predictions and collect metrics
    results_by_diagnostic: Dict[str, Any] = {}
    per_trial_rows: List[Dict[str, Any]] = []

    y_true = [valid_eval_map[t.trial_id].injected_fault for t in valid_traces]
    q_ids = [t.question_id for t in valid_traces]

    predictions_by_diag: Dict[str, Dict[str, List[Any]]] = {}

    for diag in diagnostics:
        predictions: List[str] = []
        abstained_flags: List[bool] = []
        confidences: List[float] = []
        rationales: List[str] = []

        t0 = time.perf_counter()
        for trace in valid_traces:
            d_out = diag.diagnose(trace)
            predictions.append(d_out.predicted_cause)
            abstained_flags.append(d_out.abstained)
            confidences.append(d_out.confidence)
            rationales.append(getattr(d_out, "evidence_rationale", "") or "")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        avg_ms = elapsed_ms / len(valid_traces) if valid_traces else 0.0

        predictions_by_diag[diag.name] = {
            "predictions": predictions,
            "abstained": abstained_flags,
            "confidences": confidences,
            "rationales": rationales,
        }

        # Attribution metrics
        metrics = compute_attribution_metrics(
            y_true=y_true,
            y_pred=predictions,
            labels=candidate_causes,
        )

        # Clustered bootstrap intervals
        bootstrap_summary = clustered_bootstrap_attribution(
            question_ids=q_ids,
            y_true=y_true,
            y_pred=predictions,
            labels=candidate_causes,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )

        # Identify representative misclassifications
        representative_errors = []
        for trace, yt, yp, conf, rat in zip(valid_traces, y_true, predictions, confidences, rationales):
            if yt != yp:
                representative_errors.append({
                    "trial_id": trace.trial_id,
                    "question_id": trace.question_id,
                    "question_snippet": trace.question[:80],
                    "true_cause": yt,
                    "predicted_cause": yp,
                    "confidence": round(float(conf), 3),
                    "rationale": rat,
                })

        results_by_diagnostic[diag.name] = {
            "metrics": metrics,
            "bootstrap": bootstrap_summary.model_dump(),
            "runtime": {
                "total_ms": round(elapsed_ms, 2),
                "avg_per_trial_ms": round(avg_ms, 4),
            },
            "representative_errors": representative_errors[:5],
        }

    # Build per-trial export rows using D4 predictions and all diagnostic predictions
    for idx, trace in enumerate(valid_traces):
        ev = valid_eval_map[trace.trial_id]
        pred_d4_cause = predictions_by_diag[d4.name]["predictions"][idx]
        pred_d4_abst = predictions_by_diag[d4.name]["abstained"][idx]
        pred_d4_conf = predictions_by_diag[d4.name]["confidences"][idx]
        pred_d4_rat = predictions_by_diag[d4.name]["rationales"][idx]

        row = {
            "trial_id": trace.trial_id,
            "question_id": trace.question_id,
            "true_cause": ev.injected_fault,
            "predicted_cause": pred_d4_cause,
            "abstained": pred_d4_abst,
            "confidence": pred_d4_conf,
            "rationale": pred_d4_rat,
            "pred_majority": predictions_by_diag[majority_baseline.name]["predictions"][idx],
            "pred_random": predictions_by_diag[random_baseline.name]["predictions"][idx],
            "pred_d1": predictions_by_diag[d1.name]["predictions"][idx],
            "pred_d2": predictions_by_diag[d2.name]["predictions"][idx],
            "answer": trace.answer,
        }
        if include_d3 and d3.name in predictions_by_diag:
            row["pred_d3"] = predictions_by_diag[d3.name]["predictions"][idx]
        if include_d4_learned and d4_learned.name in predictions_by_diag:
            row["pred_d4_learned"] = predictions_by_diag[d4_learned.name]["predictions"][idx]

        per_trial_rows.append(row)

    return {
        "target_split": target_split,
        "regime": regime,
        "validity_summary": validity_summary.model_dump(),
        "diagnostic_results": results_by_diagnostic,
        "trial_rows": per_trial_rows,
        "traces": traces,
        "evaluator_store": evaluator_store,
        "evaluator_records": eval_records,
    }

