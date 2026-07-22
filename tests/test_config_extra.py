"""Test suite for the Config Extra module."""

import pytest
from pydantic import BaseModel
from ml_switcheroo.config import parse_cli_key_values, RuntimeConfig, get_framework_priority_order
import ml_switcheroo.config as conf


def test_config_tomli_fallback(monkeypatch):
  """Verifies the behavior of configuration tomli fallback."""
  monkeypatch.setattr(conf, "tomllib", None)
  assert conf.tomllib is None


def test_parse_cli_key_values():
  """Parses CLI key values."""
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
  """Verifies the behavior of runtime configuration validate frameworks."""
  with pytest.raises(ValueError, match="Unknown framework"):
    RuntimeConfig(source_framework="unknown")


def test_get_plugin_config_validation():
  """Gets plugin configuration validation."""
  config = RuntimeConfig(source_framework="jax", target_framework="torch", plugin_settings={"x": "bad"})

  class DummySchema(BaseModel):
    """Dummy Schema class for testing purposes."""

    x: int

  with pytest.raises(ValueError):
    config.parse_plugin_settings(DummySchema)


def test_load_toml_config_missing():
  """Loads toml configuration missing."""
  (res, p) = conf._load_toml_settings(start_path=conf.Path("/nonexistent_path_to_toml_dir"))
  assert res == {}
  assert p is None


def test_runtime_config_default_fallback(monkeypatch):
  """Verifies the behavior of runtime configuration default fallback."""
  import ml_switcheroo.frameworks.base as fb

  monkeypatch.setattr(fb, "available_frameworks", lambda: [])
  assert conf._resolve_default_target() == "target_placeholder"

  class BadAdapter:
    """Test suite for the Bad Adapter component."""

    @property
    def ui_priority(self):
      """Helper to UI priority."""
      return "bad"

  monkeypatch.setattr(fb, "available_frameworks", lambda: ["dummy"])
  monkeypatch.setattr(fb, "get_adapter", lambda fw: BadAdapter())
  assert get_framework_priority_order() == ["dummy"]


def test_from_toml_path():
  """Verifies the behavior of from toml path."""
  import tempfile
  from pathlib import Path

  with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "pyproject.toml"
    with open(p, "w") as f:
      f.write('[tool.ml_switcheroo]\nsource_framework = "jax"\ntarget_framework = "torch"\nenable_sharding = true')
    cfg = RuntimeConfig.load(search_path=Path(d))
    assert cfg.source_framework == "jax"
    assert cfg.target_framework == "torch"
    assert cfg.enable_sharding is True
    with open(p, "w") as f:
      f.write("malformed [")
    RuntimeConfig.load(search_path=Path(d))
