"""Pydantic schemas for data loading and normalized QA representations."""

import re
from typing import Any, List, Tuple
from pydantic import BaseModel, Field, field_validator


def slugify(text: str) -> str:
    """Create a deterministic, safe identifier slug from arbitrary text."""
    slug = re.sub(r"[^\w\s-]", "", text.strip().lower())
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:50] or "unnamed"


class RawHotpotItem(BaseModel):
    """Raw record representation from HotpotQA JSON export."""
    id: str = Field(..., alias="_id")
    question: str
    answer: str
    type: str = "bridge"
    level: str = "medium"
    supporting_facts: List[Tuple[str, int]]
    context: List[Tuple[str, List[str]]]

    @field_validator("context")
    @classmethod
    def validate_context_not_empty(cls, v: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
        if not v:
            raise ValueError("Context cannot be empty.")
        return v

    @field_validator("supporting_facts")
    @classmethod
    def validate_supporting_facts(cls, v: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        if not v:
            raise ValueError("Supporting facts cannot be empty.")
        return v


class PassageRecord(BaseModel):
    """Normalized representation of a single searchable passage/document."""
    passage_id: str
    title: str
    text: str
    sentences: List[str]
    is_gold_evidence: bool = False


class QARecord(BaseModel):
    """Clean, normalized question-answering benchmark record."""
    question_id: str
    question: str
    gold_answer: str
    question_type: str
    difficulty_level: str
    gold_evidence_titles: List[str]
    gold_evidence_passage_ids: List[str]
    passages: List[PassageRecord]
