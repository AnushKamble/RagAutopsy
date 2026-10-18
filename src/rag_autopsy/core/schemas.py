"""Core data contracts and schemas for RAG-Autopsy.

Strictly separates diagnostic-visible data from privileged evaluator metadata.
"""

import hashlib
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


# Exhaustive list of forbidden substring tokens and evaluator keys
FORBIDDEN_LEAKAGE_KEYS = [
    "injected_fault",
    "injected",
    "fault",
    "fault_parameters",
    "gold_answer",
    "gold_answers",
    "gold_evidence",
    "gold_evidence_ids",
    "target_rank",
    "adjudication",
    "trial_valid",
    "invalid_reason",
    "injection_applied",
    "trace_condition_achieved",
    "clean_control",
    "evidence_unavailable",
    "evidence_ranked_out",
    "assembly_loss",
    "contradictory_evidence",
    "unsupported_generation",
    "f0",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
]

# Regex pattern matching any fault identifier token delimited by non-word chars, underscores, or bounds
FAULT_TOKEN_PATTERN = re.compile(
    r"(?:^|[\W_])(f[0-5]|fault|clean|control|hallucination|ranked_out|unavailable|assembly_loss)(?:[\W_]|$)",
    re.IGNORECASE,
)


def generate_stable_trial_id(seed: int, question_id: str, intervention_key: str, salt: str = "") -> str:
    """Generate an opaque, stable, non-label-encoding trial identifier.
    
    Produces a 16-character hexadecimal hash prefixed with 'trial_'.
    Guaranteed not to contain isolated fault label tokens or reveal intervention type.
    """
    raw = f"{seed}:{question_id}:{intervention_key}:{salt}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"trial_{digest}"


class RetrievedPassage(BaseModel):
    """A passage retrieved by the retrieval engine. Strictly forbids extra evaluator attributes."""
    model_config = ConfigDict(extra="forbid")

    passage_id: str
    rank: int
    score: float
    text: str
    title: Optional[str] = Field(default=None, description="Source article/passage title.")


class DiagnosticTrace(BaseModel):
    """The ONLY trace object visible to diagnostic methods.
    
    Contains runtime observability fields only. Must NEVER contain ground-truth
    evaluator information, injected fault IDs, or gold answer keys.
    Strictly forbids undeclared or smuggled extra fields.
    """
    model_config = ConfigDict(extra="forbid")

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
        """Strictly enforce that no privileged fields leaked into metadata, trial_id, or nested values."""
        # 1. Ensure trial_id doesn't encode fault label token (e.g. 'trial-f1-severe' or 'trial_clean')
        if FAULT_TOKEN_PATTERN.search(self.trial_id):
            raise ValueError(
                f"Leakage violation: trial_id '{self.trial_id}' encodes fault identifier!"
            )

        # 2. Check metadata dictionary for forbidden keys, string values, or smuggled evaluator objects
        for key, val in self.metadata.items():
            lower_key = str(key).lower()
            for forbidden in FORBIDDEN_LEAKAGE_KEYS:
                if forbidden in lower_key:
                    raise ValueError(
                        f"Leakage violation: forbidden key '{key}' found in DiagnosticTrace.metadata!"
                    )

            if isinstance(val, str):
                if FAULT_TOKEN_PATTERN.search(val):
                    raise ValueError(
                        f"Leakage violation: forbidden fault token in metadata value for '{key}': '{val}'!"
                    )
            elif hasattr(val, "injected_fault") or hasattr(val, "gold_answers"):
                raise ValueError(
                    f"Leakage violation: evaluator object smuggled inside DiagnosticTrace.metadata['{key}']!"
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
    Strictly forbids undeclared or smuggled extra fields.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    trial_id: str
    question_id: str
    split: str  # "dev" | "calibration" | "test"

    # Injected intervention details
    injected_fault: str  # "F0", "F1", "F2", "F3", "F4", "F5"
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
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    diagnostic_name: str
    predicted_cause: str  # "F0", "F1", "F2", "F3", "F4", "F5", "F1_F2_RETRIEVAL_MISS", "UNKNOWN"
    confidence: float
    abstained: bool = False
    abstention_reason: Optional[str] = None
    evidence_rationale: str
    runtime_ms: float


class EvaluatorStore(BaseModel):
    """Secure, encapsulated store for privileged EvaluatorRecords keyed strictly by trial_id.
    
    Guarantees that evaluator records remain decoupled from diagnostic traces and are never
    passed directly or via convenience references to diagnostic routines.
    """
    model_config = ConfigDict(extra="forbid")
    records: Dict[str, EvaluatorRecord] = Field(default_factory=dict)

    def add(self, record: EvaluatorRecord) -> None:
        """Add an evaluator record to the store, ensuring uniqueness of trial_id."""
        if not isinstance(record, EvaluatorRecord):
            raise TypeError(f"EvaluatorStore only accepts EvaluatorRecord, got {type(record).__name__}")
        if record.trial_id in self.records:
            raise ValueError(f"Duplicate trial_id in EvaluatorStore: '{record.trial_id}'")
        self.records[record.trial_id] = record

    def get(self, trial_id: str) -> Optional[EvaluatorRecord]:
        """Retrieve an evaluator record by trial_id."""
        return self.records.get(trial_id)

    def __contains__(self, trial_id: str) -> bool:
        return trial_id in self.records

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records.values())

    def to_jsonl(self, path: Union[str, Path]) -> None:
        """Safely serialize evaluator records to a dedicated evaluator-only JSONL file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for rec in self.records.values():
                f.write(rec.model_dump_json() + "\n")

    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> "EvaluatorStore":
        """Load evaluator records from a dedicated JSONL file enforcing strict schema parsing."""
        p = Path(path)
        store = cls()
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = EvaluatorRecord.model_validate_json(line)
                    store.add(rec)
        return store


class EvaluatedTrial(BaseModel):
    """Joined container produced ONLY at the post-hoc evaluation boundary.
    
    Binds a DiagnosticTrace, Diagnosis (or predictions), and EvaluatorRecord via explicit
    join on trial_id. Never visible to diagnosers; constructed exclusively for metric computation.
    """
    model_config = ConfigDict(extra="forbid")
    trial_id: str
    question_id: str
    trace: DiagnosticTrace
    evaluator_record: EvaluatorRecord
    diagnoses: Dict[str, Diagnosis] = Field(default_factory=dict)


def explicit_evaluation_join(
    traces: List[DiagnosticTrace],
    diagnoses_by_diagnostic: Dict[str, List[Diagnosis]],
    evaluator_store: EvaluatorStore,
) -> List[EvaluatedTrial]:
    """Perform strict, explicit join between visible traces, diagnostic outputs, and evaluator records.
    
    Joins exclusively on trial_id. Fails loudly if:
    1. Any trial_id in traces is missing from evaluator_store.
    2. Any diagnosis does not match a known trial_id.
    3. question_id mismatches between trace and evaluator_record.
    """
    joined: List[EvaluatedTrial] = []

    diag_lookup: Dict[str, Dict[str, Diagnosis]] = {}
    for diag_name, diag_list in diagnoses_by_diagnostic.items():
        for d in diag_list:
            if d.trial_id not in diag_lookup:
                diag_lookup[d.trial_id] = {}
            diag_lookup[d.trial_id][diag_name] = d

    for tr in traces:
        ev = evaluator_store.get(tr.trial_id)
        if ev is None:
            raise KeyError(f"Join integrity failure: trial_id '{tr.trial_id}' not found in EvaluatorStore!")
        if ev.question_id != tr.question_id:
            raise ValueError(
                f"Join integrity failure: question_id mismatch for trial_id '{tr.trial_id}' "
                f"({tr.question_id} vs {ev.question_id})!"
            )

        joined.append(
            EvaluatedTrial(
                trial_id=tr.trial_id,
                question_id=tr.question_id,
                trace=tr,
                evaluator_record=ev,
                diagnoses=diag_lookup.get(tr.trial_id, {}),
            )
        )
    return joined


def save_traces_jsonl(traces: List[DiagnosticTrace], path: Union[str, Path]) -> None:
    """Safely serialize diagnostic-visible traces to a dedicated JSONL file.
    
    Guarantees that only DiagnosticTrace schema fields are serialized.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for tr in traces:
            f.write(tr.model_dump_json() + "\n")


def load_traces_jsonl(path: Union[str, Path]) -> List[DiagnosticTrace]:
    """Load diagnostic traces from JSONL, enforcing strict schema validation (extra='forbid')."""
    p = Path(path)
    traces: List[DiagnosticTrace] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(DiagnosticTrace.model_validate_json(line))
    return traces


def save_evaluator_records_jsonl(records: List[EvaluatorRecord], path: Union[str, Path]) -> None:
    """Safely serialize evaluator records to a dedicated JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


def load_evaluator_records_jsonl(path: Union[str, Path]) -> List[EvaluatorRecord]:
    """Load evaluator records from JSONL, enforcing strict schema validation (extra='forbid')."""
    p = Path(path)
    records: List[EvaluatorRecord] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EvaluatorRecord.model_validate_json(line))
    return records

