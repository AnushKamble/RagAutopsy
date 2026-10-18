"""Unit tests for HotpotQA dataset adapter and normalization."""

from pathlib import Path
import pytest
from rag_autopsy.data.adapter import (
    create_deterministic_subset,
    load_qa_records,
    load_raw_hotpot_json,
    parse_hotpot_records,
    save_qa_records,
)
from rag_autopsy.data.schema import RawHotpotItem


def test_load_raw_hotpot_json(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    assert len(raw_items) == 2
    assert isinstance(raw_items[0], RawHotpotItem)
    assert raw_items[0].id == "syn_q001"
    assert raw_items[0].answer == "Vltava River"
    assert len(raw_items[0].context) == 10
    assert len(raw_items[0].supporting_facts) == 2


def test_parse_hotpot_records(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    qa_records = parse_hotpot_records(raw_items)

    assert len(qa_records) == 2
    rec1 = qa_records[0]
    assert rec1.question_id == "syn_q001"
    assert rec1.gold_answer == "Vltava River"
    assert rec1.gold_evidence_titles == ["Franz Kafka", "Prague"]
    assert len(rec1.gold_evidence_passage_ids) == 2
    assert len(rec1.passages) == 10

    # Verify that exactly 2 passages have is_gold_evidence == True
    gold_count = sum(1 for p in rec1.passages if p.is_gold_evidence)
    assert gold_count == 2

    # Verify passage text formatting
    kafka_passage = next(p for p in rec1.passages if p.title == "Franz Kafka")
    assert kafka_passage.is_gold_evidence is True
    assert kafka_passage.text.startswith("Franz Kafka\nFranz Kafka was a German-speaking")


def test_deterministic_subsetting(synthetic_fixture_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)

    # Sampling 1 out of 2 records
    sub1 = create_deterministic_subset(records, sample_size=1, seed=42)
    sub2 = create_deterministic_subset(records, sample_size=1, seed=42)
    assert len(sub1) == 1
    assert len(sub2) == 1
    assert sub1[0].question_id == sub2[0].question_id


def test_save_and_load_qa_records(synthetic_fixture_path: Path, tmp_path: Path) -> None:
    raw_items = load_raw_hotpot_json(synthetic_fixture_path)
    records = parse_hotpot_records(raw_items)

    out_file = tmp_path / "processed_qa.json"
    save_qa_records(records, out_file)
    assert out_file.is_file()

    loaded = load_qa_records(out_file)
    assert len(loaded) == len(records)
    assert loaded[0].question_id == records[0].question_id
    assert loaded[0].passages[0].passage_id == records[0].passages[0].passage_id


def test_load_nonexistent_dataset() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_hotpot_json("nonexistent_path_xyz.json")
