"""Config parsing: strict JSON round-trips and the flag > env > default chain."""

from dataclasses import asdict

import pytest

from bumblebee.batches import BatchPolicy
from bumblebee.config import EngineConfig, OcrConfig, env_overrides


def test_from_dict_round_trip():
    config = OcrConfig(pdf_dpi=150, force=True, ocr_request_concurrency=256)
    assert OcrConfig.from_dict(asdict(config)) == config


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="pdf_dpii"):
        OcrConfig.from_dict({"pdf_dpii": 150})


def test_from_dict_ignores_nones():
    assert OcrConfig.from_dict({"pdf_dpi": None}) == OcrConfig()


def test_defaults_are_throughput_tuned():
    config = OcrConfig()
    assert config.ocr_request_concurrency == 1024
    assert config.pdf_dpi == 100
    assert config.page_chunk_size == 8


def test_invalid_generation_values_fail_fast():
    with pytest.raises(ValueError, match="top_p"):
        OcrConfig(top_p=0.0)
    with pytest.raises(ValueError, match="max_tokens"):
        OcrConfig(max_tokens_table=0)
    with pytest.raises(ValueError, match="temperature"):
        OcrConfig(temperature=-1.0)
    with pytest.raises(ValueError, match="pdf_dpi"):
        OcrConfig(pdf_dpi=0)


def test_batch_policy_round_trip():
    policy = BatchPolicy(max_docs=8, retries=1)
    assert BatchPolicy.from_dict(asdict(policy)) == policy
    with pytest.raises(ValueError, match="max_dcos"):
        BatchPolicy.from_dict({"max_dcos": 8})


def test_env_fallback_and_explicit_precedence(monkeypatch):
    monkeypatch.setenv("BUMBLEBEE_PDF_DPI", "200")
    monkeypatch.setenv("BUMBLEBEE_MAX_NUM_SEQS", "512")
    monkeypatch.setenv("BUMBLEBEE_BATCH_RETRIES", "7")

    assert OcrConfig().pdf_dpi == 200  # env beats built-in default
    assert OcrConfig(pdf_dpi=72).pdf_dpi == 72  # explicit beats env
    engine = EngineConfig()
    assert engine.max_num_seqs == 512
    assert BatchPolicy().retries == 7


def test_empty_env_means_unset_except_speculative_config(monkeypatch):
    monkeypatch.setenv("BUMBLEBEE_LAYOUT_THRESHOLD", "")
    monkeypatch.setenv("BUMBLEBEE_SPECULATIVE_CONFIG", "")
    config = EngineConfig()
    assert config.layout_threshold is None
    assert config.speculative_config == ""  # explicit empty string disables it


def test_engine_config_env_round_trip(monkeypatch):
    config = EngineConfig(max_num_seqs=512, layout_threshold=0.4)
    for name, value in config.to_env().items():
        monkeypatch.setenv(name, value)
    assert EngineConfig() == config


def test_env_overrides_maps_set_flags_only():
    overrides = env_overrides(
        EngineConfig,
        {"vllm_extra_args": None, "gpu_memory_utilization": 0.8},
    )
    assert overrides == {"BUMBLEBEE_GPU_MEMORY_UTILIZATION": "0.8"}
    with pytest.raises(ValueError, match="modell"):
        env_overrides(EngineConfig, {"modell": "x"})


def test_engine_config_invalid_values_fail_fast():
    with pytest.raises(ValueError, match="layout_backend"):
        EngineConfig(layout_backend="tensorflow")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        EngineConfig(gpu_memory_utilization=1.5)
