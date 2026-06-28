"""Tests for docaware.config (paths, env overrides, serialization)."""

import importlib

import docaware.config as config_mod


def test_config_paths_are_under_repo():
    cfg = config_mod.AppConfig()
    assert cfg.llm.model_path.name.endswith(".gguf")
    assert "model" in str(cfg.llm.model_path).replace("\\", "/")


def test_as_dict_is_json_serializable():
    import json

    cfg = config_mod.AppConfig()
    json.dumps(cfg.as_dict())  # must not raise (paths rendered as strings)


def test_env_override(monkeypatch):
    monkeypatch.setenv("ADTC_N_CTX", "2048")
    importlib.reload(config_mod)
    assert config_mod.LLMConfig().n_ctx == 2048
    monkeypatch.delenv("ADTC_N_CTX")
    importlib.reload(config_mod)
