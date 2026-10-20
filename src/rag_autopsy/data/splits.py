"""Deterministic question-grouped split generation and manifest tracking."""

import random
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from rag_autopsy.data.schema import QARecord


class SplitManifest(BaseModel):
    """Manifest recording the exact partition of question IDs across splits."""
    manifest_version: str = "1.0.0"
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    random_seed: int
    total_questions: int
    split_counts: dict[str, int]
    dev_ids: list[str]
    calibration_ids: list[str]
    test_ids: list[str]

    @model_validator(mode="after")
    def verify_split_disjointness(self) -> "SplitManifest":
        """Verify that dev, calibration, and test sets are strictly pairwise disjoint."""
        s_dev: set[str] = set(self.dev_ids)
        s_cal: set[str] = set(self.calibration_ids)
        s_test: set[str] = set(self.test_ids)

        if s_dev.intersection(s_cal):
            overlap = s_dev.intersection(s_cal)
            raise ValueError(f"Data split contamination: dev and calibration overlap on {len(overlap)} IDs!")

        if s_dev.intersection(s_test):
            overlap = s_dev.intersection(s_test)
            raise ValueError(f"Data split contamination: dev and test overlap on {len(overlap)} IDs!")

        if s_cal.intersection(s_test):
            overlap = s_cal.intersection(s_test)
            raise ValueError(f"Data split contamination: calibration and test overlap on {len(overlap)} IDs!")

        expected_total = len(self.dev_ids) + len(self.calibration_ids) + len(self.test_ids)
        if self.total_questions != expected_total:
            raise ValueError(
                f"Total questions mismatch: total_questions={self.total_questions} vs sum={expected_total}"
            )

        return self

    def get_split_for_question(self, question_id: str) -> str:
        """Return the partition name ('dev', 'calibration', or 'test') for a question_id."""
        if question_id in self.dev_ids:
            return "dev"
        elif question_id in self.calibration_ids:
            return "calibration"
        elif question_id in self.test_ids:
            return "test"
        raise KeyError(f"Question ID '{question_id}' not found in any split manifest partition!")


def create_question_splits(
    records: list[QARecord],
    dev_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> SplitManifest:
    """Deterministically partition records into dev, calibration, and test splits."""
    if abs((dev_ratio + calibration_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got: dev={dev_ratio}, cal={calibration_ratio}, test={test_ratio}"
        )

    # Sort question IDs for deterministic cross-platform ordering before shuffle
    question_ids = sorted(list(set(r.question_id for r in records)))
    n = len(question_ids)

    rng = random.Random(seed)
    shuffled_ids = list(question_ids)
    rng.shuffle(shuffled_ids)

    n_dev = int(round(n * dev_ratio))
    n_cal = int(round(n * calibration_ratio))
    # Test gets the remainder to guarantee sum == n
    n_test = n - (n_dev + n_cal)

    dev_ids = shuffled_ids[:n_dev]
    calibration_ids = shuffled_ids[n_dev : n_dev + n_cal]
    test_ids = shuffled_ids[n_dev + n_cal :]

    manifest = SplitManifest(
        random_seed=seed,
        total_questions=n,
        split_counts={
            "dev": len(dev_ids),
            "calibration": len(calibration_ids),
            "test": len(test_ids),
        },
        dev_ids=dev_ids,
        calibration_ids=calibration_ids,
        test_ids=test_ids,
    )
    return manifest


def save_split_manifest(manifest: SplitManifest, output_path: str | Path) -> None:
    """Save SplitManifest to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))


def load_split_manifest(file_path: str | Path) -> SplitManifest:
    """Load SplitManifest from JSON."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        return SplitManifest.model_validate_json(f.read())
