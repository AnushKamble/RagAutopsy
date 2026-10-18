"""In-memory corpus representation and trial-scoped view management."""

from typing import Dict, Iterable, List, Optional, Set
from rag_autopsy.data.schema import PassageRecord, QARecord


class Corpus:
    """In-memory collection of searchable passages with immutable trial views."""

    def __init__(self, passages: Optional[List[PassageRecord]] = None):
        self._passages: List[PassageRecord] = list(passages or [])
        self._by_id: Dict[str, PassageRecord] = {p.passage_id: p for p in self._passages}

    @classmethod
    def from_qa_records(cls, records: Iterable[QARecord]) -> "Corpus":
        """Construct corpus from all passages across the provided QA records."""
        seen_ids: Set[str] = set()
        unique_passages: List[PassageRecord] = []

        for record in records:
            for p in record.passages:
                if p.passage_id not in seen_ids:
                    seen_ids.add(p.passage_id)
                    unique_passages.append(p)

        return cls(unique_passages)

    def get_passage(self, passage_id: str) -> Optional[PassageRecord]:
        """Look up a passage by its unique identifier."""
        return self._by_id.get(passage_id)

    @property
    def passages(self) -> List[PassageRecord]:
        """Return all passages in this corpus view."""
        return list(self._passages)

    def __len__(self) -> int:
        return len(self._passages)

    def without_passages(self, exclude_ids: Set[str]) -> "Corpus":
        """Create a trial-scoped view excluding designated passage IDs (used for F1 injection).
        
        Guarantees that the underlying shared corpus is never permanently mutated.
        """
        filtered = [p for p in self._passages if p.passage_id not in exclude_ids]
        return Corpus(filtered)

    def with_additional_passages(self, new_passages: List[PassageRecord]) -> "Corpus":
        """Create a trial-scoped view with additional passages (used for F2/F4 injection)."""
        combined = list(self._passages) + list(new_passages)
        return Corpus(combined)
