"""Data ingestion, schema validation, and question-grouped partitioning."""

from rag_autopsy.data.schema import RawHotpotItem, QARecord, PassageRecord
from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.data.splits import create_question_splits, SplitManifest

__all__ = [
    "RawHotpotItem",
    "QARecord",
    "PassageRecord",
    "load_raw_hotpot_json",
    "parse_hotpot_records",
    "create_question_splits",
    "SplitManifest",
]
