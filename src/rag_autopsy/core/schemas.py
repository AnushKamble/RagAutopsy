"""Core data contracts and schemas for RAG-Autopsy.

Strictly separates diagnostic-visible data from privileged evaluator metadata.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator


# List of forbidden substring tokens in diagnostic-visible fields to prevent leakage
FORBIDDEN_LEAKAGE_KEYS = [
    "injected_fault",
    "injected",
    "gold_answer",
    "gold_evidence",
    "gold_evidence_ids",
    "target_rank",
    "adjudication",
    "trial_valid",
]


class RetrievedPassage(BaseModel):
    """A passage retrieved by the retrieval engine."""
    passage_id: str
    rank: int
    score: float
    text: str
    title: Optional[str] = Field(default=None, description="Source article/passage title.")


class DiagnosticTrace(BaseModel):
    """The ONLY trace object visible to diagnostic methods.
    
    Contains runtime observability fields only. Must NEVER contain ground-truth
    evaluator information, injected fault IDs, or gold answer keys.
    """
    schema_version: str = "1.0.0"
    trial_id: str = Field(..., description="Opaque trial UUID. Must NEVER encode fault type.")
    question_id: str = Field(..., description="Stable question identifier.")
    question: str

    # Retrieval pre-assembly trace
    retrieved_top_k: List[RetrievedPassage]
    candidate_pool_scores: Optional[List[float]] = Field(
        default=None,
        description="Retriever scores for top-K (e.g. K=20) candidates to detect score cliffs without leaking text."
    )
    source_passage_ids: List[str] = Field(
        default_factory=list,
        description="Explicit source passage IDs retrieved from corpus."
    )

    # Context assembly trace
    pre_assembly_passage_ids: List[str]
    post_assembly_passage_ids: List[str]
    generator_context: str

    # Generation trace
    answer: str
    latency_ms: float
    generator_model: Optional[str] = Field(default=None, description="Identifier of generator model used.")
    generator_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters of generator.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def verify_no_leakage(self) -> "DiagnosticTrace":
        """Strictly enforce that no privileged fields leaked into metadata or trial_id."""
        # Ensure trial_id doesn't encode fault label (e.g. 'f1-test')
        lower_trial_id = self.trial_id.lower()
        for forbidden in ["f1", "f2", "f3", "f4", "fault"]:
            if f"-{forbidden}-" in lower_trial_id or lower_trial_id.startswith(f"{forbidden}_"):
                raise ValueError(
                    f"Leakage violation: trial_id '{self.trial_id}' encodes fault identifier '{forbidden}'!"
                )

        # Check metadata dictionary for forbidden keys
        for key in self.metadata.keys():
            lower_key = key.lower()
            for forbidden in FORBIDDEN_LEAKAGE_KEYS:
                if forbidden in lower_key:
                    raise ValueError(
                        f"Leakage violation: forbidden key '{key}' found in DiagnosticTrace.metadata!"
                    )
        return self


def verify_trace_reconstructibility(trace: DiagnosticTrace) -> Tuple[bool, Optional[str]]:
    """Validate that the trace contains complete information to reconstruct what the generator saw.
    
    Verifies that:
    1. Every post-assembly passage ID corresponds to an observable retrieved passage in retrieved_top_k.
    2. The text of every surviving passage is contained within generator_context.
    3. The generator context is non-empty whenever post_assembly_passage_ids is non-empty.
    
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str]).
    """
    retrieved_map = {p.passage_id: p for p in trace.retrieved_top_k}

    if trace.post_assembly_passage_ids and not trace.generator_context.strip():
        return False, "Non-empty post-assembly passage IDs but generator context is blank."

    for pid in trace.post_assembly_passage_ids:
        if pid not in retrieved_map:
            return False, f"Post-assembly passage ID '{pid}' was not present in retrieved_top_k."
        p = retrieved_map[pid]
        # Verify text of the passage is present in generator_context
        # Check either full text or a substantial normalized snippet
        snippet = p.text[:min(100, len(p.text))].strip()
        if snippet and snippet not in trace.generator_context:
            return False, f"Text of passage '{pid}' is missing from generator_context."

    return True, None


class EvaluatorRecord(BaseModel):
    """Privileged record accessible ONLY to the evaluator.
    
    Contains ground-truth intervention parameters, gold labels, and validity adjudication.
    """
    schema_version: str = "1.0.0"
    trial_id: str
    question_id: str
    split: str  # "dev" | "calibration" | "test"

    # Injected intervention details
    injected_fault: str  # "F0", "F1", "F2", "F3", "F4"
    fault_parameters: Dict[str, Any] = Field(default_factory=dict)

    # Gold reference labels
    gold_answers: List[str]
    gold_evidence_ids: List[str]

    # Injection validity adjudication
    injection_applied: bool = True
    trace_condition_achieved: bool = True
    trial_valid: bool = True
    invalid_reason: Optional[str] = None

    # Downstream evaluation results
    answer_exact_match: Optional[bool] = None
    answer_token_f1: Optional[float] = None
    answer_supported_by_context: Optional[bool] = None


class Diagnosis(BaseModel):
    """Output produced by a diagnostic method or attribution policy."""
    trial_id: str
    diagnostic_name: str
    predicted_cause: str  # "F0", "F1", "F2", "F3", "F4", "F1_F2_RETRIEVAL_MISS", "UNKNOWN"
    confidence: float
    abstained: bool = False
    abstention_reason: Optional[str] = None
    evidence_rationale: str
    runtime_ms: float
