"""Core data contracts and schemas for RAG-Autopsy.

Strictly separates diagnostic-visible data from privileged evaluator metadata.
"""

from typing import Any, Dict, List, Optional
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

    # Context assembly trace
    pre_assembly_passage_ids: List[str]
    post_assembly_passage_ids: List[str]
    generator_context: str

    # Generation trace
    answer: str
    latency_ms: float
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
