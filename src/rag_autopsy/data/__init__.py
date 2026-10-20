"""Data ingestion, schema validation, and question-grouped partitioning."""

from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.data.schema import PassageRecord, QARecord, RawHotpotItem
from rag_autopsy.data.splits import SplitManifest, create_question_splits

__all__ = [
    "PassageRecord",
    "QARecord",
    "RawHotpotItem",
    "SplitManifest",
    "create_question_splits",
    "load_raw_hotpot_json",
    "parse_hotpot_records",
]
