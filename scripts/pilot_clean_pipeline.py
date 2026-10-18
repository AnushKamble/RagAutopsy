"""Pilot evaluation script for the clean RAG reference pipeline on HotpotQA."""

import json
from pathlib import Path
from typing import Any, Dict, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_autopsy.core.schemas import verify_trace_reconstructibility
from rag_autopsy.data.adapter import load_qa_records
from rag_autopsy.evaluation.answer_metrics import compute_exact_match, compute_token_f1
from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import CachedGenerator, MockExtractiveGenerator
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import BM25Retriever


def run_clean_pipeline_pilot(
    data_path: str = ".data/processed/hotpot_subset_100.json",
    manifest_path: str = "configs/split_manifest_100.json",
    split: str = "test",
    top_k: int = 3,
    output_path: str = "results/clean_pilot_results.json",
) -> Dict[str, Any]:
    console = Console()
    console.print(Panel(f"[bold cyan]Running Clean RAG Reference Pipeline Pilot ({split.upper()})[/bold cyan]"))

    records = load_qa_records(data_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_ids = set(manifest[f"{split}_ids"])
    target_records = [r for r in records if r.question_id in target_ids]
    console.print(f"Loaded [cyan]{len(target_records)}[/cyan] clean questions for target split '{split}'.")

    corpus = Corpus.from_qa_records(target_records)
    retriever = BM25Retriever(corpus)
    builder = ContextBuilder(max_context_chars=3000)
    generator = CachedGenerator(MockExtractiveGenerator(), cache_dir=".cache/pilot_gen")

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=builder,
        generator=generator,
        top_k=top_k,
        candidate_pool_size=20,
    )

    trials: List[Dict[str, Any]] = []
    reconstruction_passes = 0

    for idx, rec in enumerate(target_records, start=1):
        trace, eval_rec = pipeline.run(
            trial_id=f"clean_pilot_{idx:03d}",
            question_id=rec.question_id,
            question=rec.question,
            gold_answers=[rec.gold_answer],
            gold_evidence_ids=rec.gold_evidence_passage_ids,
            injected_fault="F0",
        )

        # Audit trace reconstructibility
        is_recon, err = verify_trace_reconstructibility(trace)
        if is_recon:
            reconstruction_passes += 1

        # Evaluate Answer Correctness
        em = compute_exact_match(trace.answer, rec.gold_answer)
        f1 = compute_token_f1(trace.answer, rec.gold_answer)

        # Evaluate Grounding & Support
        retrieved_ids = set(trace.source_passage_ids)
        gold_ids = set(rec.gold_evidence_passage_ids)
        gold_retrieved_count = len(retrieved_ids.intersection(gold_ids))
        any_gold_retrieved = gold_retrieved_count > 0
        all_gold_retrieved = gold_ids.issubset(retrieved_ids)

        # Context support: check if reference answer appears in assembled context
        answer_in_context = rec.gold_answer.lower() in trace.generator_context.lower()

        trials.append({
            "trial_id": trace.trial_id,
            "question_id": rec.question_id,
            "question": rec.question,
            "gold_answer": rec.gold_answer,
            "gold_evidence_passage_ids": rec.gold_evidence_passage_ids,
            "retrieved_passage_ids": trace.source_passage_ids,
            "retrieved_top_k_ranks_scores": [(p.rank, p.score, p.title) for p in trace.retrieved_top_k],
            "assembled_context_char_length": len(trace.generator_context),
            "generated_answer": trace.answer,
            "exact_match": em,
            "token_f1": round(f1, 4),
            "any_gold_retrieved": any_gold_retrieved,
            "all_gold_retrieved": all_gold_retrieved,
            "gold_retrieved_count": gold_retrieved_count,
            "total_gold_count": len(gold_ids),
            "answer_in_context": answer_in_context,
            "trace_reconstructible": is_recon,
            "reconstruction_error": err,
            "latency_ms": trace.latency_ms,
        })

    # Aggregate Metrics
    n = len(trials)
    em_rate = sum(1 for t in trials if t["exact_match"]) / n
    avg_f1 = sum(t["token_f1"] for t in trials) / n
    any_recall = sum(1 for t in trials if t["any_gold_retrieved"]) / n
    all_recall = sum(1 for t in trials if t["all_gold_retrieved"]) / n
    context_support_rate = sum(1 for t in trials if t["answer_in_context"]) / n
    recon_rate = reconstruction_passes / n

    summary = {
        "split": split,
        "sample_size": n,
        "answer_correctness": {
            "exact_match_rate": round(em_rate, 4),
            "mean_token_f1": round(avg_f1, 4),
        },
        "support_and_grounding": {
            "context_support_rate": round(context_support_rate, 4),
            "any_gold_evidence_recall": round(any_recall, 4),
            "all_gold_evidence_recall": round(all_recall, 4),
        },
        "instrumentation_and_trace_health": {
            "trace_reconstructibility_rate": round(recon_rate, 4),
            "mean_latency_ms": round(sum(t["latency_ms"] for t in trials) / n, 2),
        },
    }

    # Print Summary Tables
    tab_correctness = Table(title=f"Answer Correctness Metrics (N={n})")
    tab_correctness.add_column("Metric", style="bold yellow")
    tab_correctness.add_column("Value", style="cyan")
    tab_correctness.add_row("Exact Match (EM)", f"{em_rate*100:.1f}%")
    tab_correctness.add_row("Token F1", f"{avg_f1:.3f}")
    console.print(tab_correctness)

    tab_grounding = Table(title=f"Retrieval Support & Grounding Metrics (N={n}, top_k={top_k})")
    tab_grounding.add_column("Metric", style="bold yellow")
    tab_grounding.add_column("Value", style="green")
    tab_grounding.add_row("Any Gold Evidence Retrieved (Recall@3)", f"{any_recall*100:.1f}%")
    tab_grounding.add_row("All Gold Evidence Retrieved (2-Hop Recall@3)", f"{all_recall*100:.1f}%")
    tab_grounding.add_row("Context Support (Answer in Context)", f"{context_support_rate*100:.1f}%")
    tab_grounding.add_row("Trace Reconstructibility Audit", f"{recon_rate*100:.1f}%")
    console.print(tab_grounding)

    # Manual Inspection of Representative Samples
    console.print(Panel("[bold]Manual Inspection of Representative Clean Trials[/bold]"))
    for t in trials[:4]:
        clean_q = t['question'].encode('ascii', errors='replace').decode('ascii')
        clean_ans = t['generated_answer'][:120].encode('ascii', errors='replace').decode('ascii')
        clean_top3 = [(r, s, tit.encode('ascii', errors='replace').decode('ascii') if tit else None) for r, s, tit in t['retrieved_top_k_ranks_scores']]
        console.print(f"[bold cyan]Trial ID:[/bold cyan] {t['trial_id']} | [bold]Question:[/bold] \"{clean_q}\"")
        console.print(f"  [dim]Gold Answer:[/dim] [yellow]\"{t['gold_answer']}\"[/yellow]")
        console.print(f"  [dim]Retrieved Top-3:[/dim] {clean_top3}")
        console.print(f"  [dim]Generated Answer:[/dim] \"{clean_ans}...\"")
        console.print(
            f"  [bold]Correctness:[/bold] EM={t['exact_match']} (F1={t['token_f1']}) | "
            f"[bold]Grounding:[/bold] AnyGold={t['any_gold_retrieved']} | AllGold={t['all_gold_retrieved']} | "
            f"AnswerInContext={t['answer_in_context']}\n"
        )

    # Save artifact
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "trials": trials}, f, indent=2)

    console.print(f"Saved full clean pilot report: [bold green]{output_path}[/bold green]")
    return summary
