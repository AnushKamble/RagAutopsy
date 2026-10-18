"""Configuration loading and validation using Pydantic."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str = "hotpotqa"
    split: str = "dev"
    setting: str = "distractor"
    data_dir: str = ".data"
    raw_file: Optional[str] = None
    sample_size: int = 500
    seed: int = 42
    train_ratio: float = 0.6
    calibration_ratio: float = 0.2
    test_ratio: float = 0.2


class RetrieverConfig(BaseModel):
    backend: str = "bm25"
    top_k: int = 3
    candidate_pool_size: int = 20
    k1: float = 1.5
    b: float = 0.75


class GeneratorConfig(BaseModel):
    backend: str = "mock"  # "mock" | "hf_local" | "api"
    model_name: str = "extractive_baseline"
    max_tokens: int = 64
    temperature: float = 0.0
    cache_dir: str = ".cache/generator"


class DiagnosticConfig(BaseModel):
    enabled_diagnostics: List[str] = Field(default_factory=lambda: ["d1_retrieval", "d2_lexical"])
    abstention_enabled: bool = True
    regime: str = "regime_a"  # "regime_a" (candidate pool probe) | "regime_b" (blind top-k)
    d1_score_gap_threshold: float = 0.5
    d2_overlap_threshold: float = 0.2


class ExperimentConfig(BaseModel):
    experiment_name: str = "pilot_run"
    output_dir: str = "results"
    random_seed: int = 42
    bootstrap_resamples: int = 1000
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    diagnostics: DiagnosticConfig = Field(default_factory=DiagnosticConfig)


def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """Load configuration from a YAML file, merging with default values."""
    if config_path is None:
        return ExperimentConfig()

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return ExperimentConfig.model_validate(data)


def save_config(config: ExperimentConfig, output_path: str) -> None:
    """Save an ExperimentConfig instance to a YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False)
