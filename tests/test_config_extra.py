import pytest
import sys
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, ValidationError
from ml_switcheroo.config import parse_cli_key_values, RuntimeConfig, get_framework_priority_order
import ml_switcheroo.config as conf


def test_config_tomli_fallback(monkeypatch):
  monkeypatch.setattr(conf, "tomllib", None)
  assert conf.tomllib is None


def test_parse_cli_key_values():
  assert parse_cli_key_values(None) == {}
  assert parse_cli_key_values([]) == {}

  items = [
    "invalid_format",
    "key1=value1",
    "key_true=True",
    "key_false=fAlse",
    "key_int=42",
    "key_float=3.14",
    "key_exp=1e-5",
    "key_str_num=42a",
  ]
  res = parse_cli_key_values(items)
  assert res["key1"] == "value1"


def test_runtime_config_validate_frameworks():
  with pytest.raises(ValueError, match="Unknown framework"):
    RuntimeConfig(source_framework="unknown")


def test_get_plugin_config_validation():
  config = RuntimeConfig(source_framework="jax", target_framework="torch", plugin_settings={"x": "bad"})

  class DummySchema(BaseModel):
    x: int

  with pytest.raises(ValueError):
    config.parse_plugin_settings(DummySchema)


def test_load_toml_config_missing():
  res, p = conf._load_toml_settings(start_path=conf.Path("/nonexistent_path_to_toml_dir"))
  assert res == {}
  assert p is None


def test_runtime_config_default_fallback(monkeypatch):
  import ml_switcheroo.frameworks.base as fb

  monkeypatch.setattr(fb, "available_frameworks", lambda: [])

  assert conf._resolve_default_target() == "target_placeholder"

  class BadAdapter:
    @property
    def ui_priority(self):
      return "bad"

  monkeypatch.setattr(fb, "available_frameworks", lambda: ["dummy"])
  monkeypatch.setattr(fb, "get_adapter", lambda fw: BadAdapter())

  assert get_framework_priority_order() == ["dummy"]


def test_from_toml_path():
  import tempfile
  import json
  from pathlib import Path

  with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "pyproject.toml"
    with open(p, "w") as f:
      f.write('[tool.ml_switcheroo]\nsource_framework = "jax"\ntarget_framework = "torch"\nenable_sharding = true')

    cfg = RuntimeConfig.load(search_path=Path(d))
    assert cfg.source_framework == "jax"
    assert cfg.target_framework == "torch"
    assert cfg.enable_sharding is True

    # 336-338: Malformed TOML
    with open(p, "w") as f:
      f.write("malformed [")
    cfg2 = RuntimeConfig.load(search_path=Path(d))
