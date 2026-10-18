"""Unit tests for question-grouped split generation and manifest disjointness."""

from pathlib import Path
import pytest
from rag_autopsy.data.adapter import load_raw_hotpot_json, parse_hotpot_records
from rag_autopsy.data.schema import QARecord
from rag_autopsy.data.splits import (
    SplitManifest,
    create_question_splits,
    load_split_manifest,
    save_split_manifest,
)


def _generate_dummy_records(n: int) -> list[QARecord]:
    """Helper to generate n dummy QARecords for split testing."""
    return [
        QARecord(
            question_id=f"q_{i:04d}",
            question=f"Question {i}?",
            gold_answer=f"Answer {i}",
            question_type="bridge",
            difficulty_level="medium",
            gold_evidence_titles=["TitleA"],
            gold_evidence_passage_ids=["pA"],
            passages=[],
        )
        for i in range(n)
    ]


def test_split_generation_and_ratios() -> None:
    records = _generate_dummy_records(100)
    manifest = create_question_splits(records, dev_ratio=0.6, calibration_ratio=0.2, test_ratio=0.2, seed=42)

    assert manifest.total_questions == 100
    assert len(manifest.dev_ids) == 60
    assert len(manifest.calibration_ids) == 20
    assert len(manifest.test_ids) == 20
    assert manifest.split_counts == {"dev": 60, "calibration": 20, "test": 20}


def test_split_disjointness_invariant() -> None:
    records = _generate_dummy_records(50)
    manifest = create_question_splits(records, dev_ratio=0.6, calibration_ratio=0.2, test_ratio=0.2, seed=123)

    s_dev = set(manifest.dev_ids)
    s_cal = set(manifest.calibration_ids)
    s_test = set(manifest.test_ids)

    # Strictly 0 intersection between any pair
    assert s_dev.intersection(s_cal) == set()
    assert s_dev.intersection(s_test) == set()
    assert s_cal.intersection(s_test) == set()

    # Partition covers exactly all items
    all_split_ids = s_dev.union(s_cal).union(s_test)
    assert len(all_split_ids) == 50


def test_get_split_for_question() -> None:
    records = _generate_dummy_records(10)
    manifest = create_question_splits(records, dev_ratio=0.6, calibration_ratio=0.2, test_ratio=0.2, seed=99)

    first_dev_id = manifest.dev_ids[0]
    assert manifest.get_split_for_question(first_dev_id) == "dev"

    first_cal_id = manifest.calibration_ids[0]
    assert manifest.get_split_for_question(first_cal_id) == "calibration"

    first_test_id = manifest.test_ids[0]
    assert manifest.get_split_for_question(first_test_id) == "test"

    with pytest.raises(KeyError):
        manifest.get_split_for_question("nonexistent_id_9999")


def test_manifest_contamination_rejection() -> None:
    """Negative test: if an ID is artificially injected into both dev and test, validation must fail."""
    with pytest.raises(ValueError, match="Data split contamination"):
        SplitManifest(
            random_seed=42,
            total_questions=3,
            split_counts={"dev": 2, "calibration": 0, "test": 1},
            dev_ids=["q1", "q2"],
            calibration_ids=[],
            test_ids=["q2"],  # Contamination!
        )


def test_manifest_save_and_load(tmp_path: Path) -> None:
    records = _generate_dummy_records(20)
    manifest = create_question_splits(records, seed=777)

    manifest_path = tmp_path / "splits.json"
    save_split_manifest(manifest, manifest_path)
    assert manifest_path.is_file()

    loaded = load_split_manifest(manifest_path)
    assert loaded.random_seed == 777
    assert loaded.dev_ids == manifest.dev_ids
    assert loaded.test_ids == manifest.test_ids
