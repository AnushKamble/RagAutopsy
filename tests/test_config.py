"""Tests for configuration parsing and validation."""

from pathlib import Path

import pytest

from rag_autopsy.core.config import ExperimentConfig, load_config, save_config


def test_default_config() -> None:
    cfg = ExperimentConfig()
    assert cfg.experiment_name == "pilot_run"
    assert cfg.dataset.name == "hotpotqa"
    assert cfg.retriever.backend == "bm25"
    assert cfg.retriever.top_k == 3
    assert cfg.retriever.candidate_pool_size == 20
    assert cfg.diagnostics.regime == "regime_a"
    assert cfg.diagnostics.abstention_enabled is True


def test_config_save_and_load(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        experiment_name="custom_test_experiment",
        random_seed=12345,
    )
    cfg.retriever.top_k = 5
    config_file = tmp_path / "test_config.yaml"

    save_config(cfg, str(config_file))
    assert config_file.is_file()

    loaded = load_config(str(config_file))
    assert loaded.experiment_name == "custom_test_experiment"
    assert loaded.random_seed == 12345
    assert loaded.retriever.top_k == 5
    assert loaded.dataset.name == "hotpotqa"


def test_load_nonexistent_config() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config_path_12345.yaml")
