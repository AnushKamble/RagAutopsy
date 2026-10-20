"""Dataset adapter for HotpotQA: loading, parsing, and normalization."""

import json
import random
from pathlib import Path

from pydantic import ValidationError

from rag_autopsy.data.schema import PassageRecord, QARecord, RawHotpotItem, slugify


def load_raw_hotpot_json(file_path: str | Path) -> list[RawHotpotItem]:
    """Load and validate raw HotpotQA JSON file into RawHotpotItem models."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"HotpotQA data file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array of records, got {type(data).__name__}")

    items: list[RawHotpotItem] = []
    errors = []
    for idx, raw_dict in enumerate(data):
        try:
            items.append(RawHotpotItem.model_validate(raw_dict))
        except ValidationError as e:
            errors.append(f"Record at index {idx} failed validation: {e}")

    if errors:
        raise ValueError(f"Failed to load {len(errors)} records from {file_path}:\n" + "\n".join(errors[:5]))

    return items


def parse_hotpot_records(raw_items: list[RawHotpotItem]) -> list[QARecord]:
    """Convert raw HotpotQA items into normalized QARecord domain objects."""
    records: list[QARecord] = []

    for item in raw_items:
        # Extract unique gold evidence titles
        gold_titles = list(dict.fromkeys(title for title, _ in item.supporting_facts))

        passages: list[PassageRecord] = []
        gold_passage_ids: list[str] = []

        for p_idx, (title, sentences) in enumerate(item.context):
            is_gold = title in gold_titles
            # Deterministic, unique passage identifier combining question id, index, and slugified title
            passage_id = f"{item.id}__p{p_idx}_{slugify(title)}"
            passage_text = f"{title}\n" + " ".join(sentences)

            passage = PassageRecord(
                passage_id=passage_id,
                title=title,
                text=passage_text,
                sentences=sentences,
                is_gold_evidence=is_gold,
            )
            passages.append(passage)

            if is_gold:
                gold_passage_ids.append(passage_id)

        records.append(
            QARecord(
                question_id=item.id,
                question=item.question.strip(),
                gold_answer=item.answer.strip(),
                question_type=item.type,
                difficulty_level=item.level,
                gold_evidence_titles=gold_titles,
                gold_evidence_passage_ids=gold_passage_ids,
                passages=passages,
            )
        )

    return records


def create_deterministic_subset(
    records: list[QARecord], sample_size: int, seed: int = 42
) -> list[QARecord]:
    """Select a deterministic subset of records using a fixed random seed."""
    if sample_size >= len(records):
        return list(records)

    rng = random.Random(seed)
    # Sort first by question_id to guarantee identical ordering across platforms
    sorted_records = sorted(records, key=lambda r: r.question_id)
    return rng.sample(sorted_records, sample_size)


def save_qa_records(records: list[QARecord], output_path: str | Path) -> None:
    """Save normalized QARecord objects to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in records], f, indent=2)


def load_qa_records(file_path: str | Path) -> list[QARecord]:
    """Load normalized QARecord objects from a JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"QA records file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    return [QARecord.model_validate(item) for item in raw_list]


def load_raw_hotpot_parquet(
    file_path: str | Path, limit: int | None = None
) -> list[RawHotpotItem]:
    """Load and validate raw HotpotQA records from a Parquet file (e.g. from Hugging Face)."""
    import pyarrow.parquet as pq

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    table = pq.read_table(str(path))
    if limit is not None and limit < table.num_rows:
        table = table.slice(0, limit)

    rows = table.to_pylist()
    items: list[RawHotpotItem] = []

    for row in rows:
        sf = list(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]))
        ctx = list(zip(row["context"]["title"], row["context"]["sentences"]))
        items.append(
            RawHotpotItem(
                _id=row["id"],
                question=row["question"],
                answer=row["answer"],
                type=row["type"],
                level=row["level"],
                supporting_facts=sf,
                context=ctx,
            )
        )

    return items
