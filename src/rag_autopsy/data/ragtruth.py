"""RAGTruth dataset adapter and schema for hallucination/grounding benchmarking.

Spec Section 5, 8, 23:
RAGTruth provides human-annotated RAG responses with word/span-level hallucination labels.
This module adapts RAGTruth items into the RAG-Autopsy evaluation harness to benchmark
whether D2 (Lexical Support) and D3 (NLI Support) agree with human hallucination judgments.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
from rag_autopsy.core.schemas import DiagnosticTrace, RetrievedPassage, generate_stable_trial_id


class RAGTruthSample(BaseModel):
    """A human-annotated sample from the RAGTruth grounding benchmark."""
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    prompt: str
    source_passage: str
    response: str
    has_hallucination: bool = Field(..., description="True if response contains human-annotated hallucination.")
    hallucination_spans: List[str] = Field(default_factory=list)
    hallucination_type: Optional[str] = Field(
        default=None,
        description="Type: evidential_conflict, unsupported_fact, or none."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_diagnostic_trace(self, seed: int = 42) -> DiagnosticTrace:
        """Convert to a strictly isolated DiagnosticTrace visible to diagnostics.
        
        Evaluator-only fields (has_hallucination, hallucination_spans) are strictly withheld.
        """
        opaque_id = generate_stable_trial_id(seed=seed, question_id=self.sample_id, intervention_key="rt_sample")
        src_id = f"rt_doc_{self.sample_id}"

        passage = RetrievedPassage(
            passage_id=src_id,
            rank=1,
            score=10.0,
            text=self.source_passage,
            title=f"Source Doc {self.sample_id}",
        )

        return DiagnosticTrace(
            trial_id=opaque_id,
            question_id=self.sample_id,
            question=self.prompt,
            retrieved_top_k=[passage],
            candidate_pool_scores=[10.0],
            source_passage_ids=[src_id],
            pre_assembly_passage_ids=[src_id],
            post_assembly_passage_ids=[src_id],
            generator_context=self.source_passage,
            answer=self.response,
            latency_ms=15.0,
            generator_model="annotated_model",
            generator_config={},
            metadata={"dataset": "ragtruth"},
        )


def load_ragtruth_json(file_path: Union[str, Path]) -> List[RAGTruthSample]:
    """Load and parse RAGTruth samples from a JSON file."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"RAGTruth file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of RAGTruth samples in {file_path}, got {type(data)}")

    samples = [RAGTruthSample.model_validate(item) for item in data]
    return samples
