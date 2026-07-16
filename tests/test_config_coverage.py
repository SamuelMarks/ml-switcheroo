"""Auto-generated doc."""

import pytest
from pydantic import BaseModel, Field

from ml_switcheroo.config import (
  RuntimeConfig,
  _resolve_default_source,
  _resolve_default_target,
  parse_cli_key_values,
  _load_toml_settings,
)
from unittest.mock import patch


def test_resolve_default_placeholders():
  """Auto-generated doc."""
  with patch("ml_switcheroo.config.get_framework_priority_order", return_value=[]):
    assert _resolve_default_source() == "source_placeholder"
    assert _resolve_default_target() == "target_placeholder"

  with patch("ml_switcheroo.config.get_framework_priority_order", return_value=["one"]):
    assert _resolve_default_source() == "one"
    assert _resolve_default_target() == "one"


def test_effective_frameworks():
  """Auto-generated doc."""
  config = RuntimeConfig(
    source_framework="torch", target_framework="jax", source_flavour="torch_flavour", target_flavour="jax_flavour"
  )
  assert config.effective_source == "torch_flavour"
  assert config.effective_target == "jax_flavour"

  config2 = RuntimeConfig(source_framework="torch", target_framework="jax")
  assert config2.effective_source == "torch"
  assert config2.effective_target == "jax"


class DummySchema(BaseModel):
  """Auto-generated doc."""

  my_int: int = Field(...)


def test_parse_plugin_settings_error():
  """Auto-generated doc."""
  config = RuntimeConfig(plugin_settings={"my_int": "not_an_int"})
  with pytest.raises(ValueError, match="Plugin configuration validation failed"):
    config.parse_plugin_settings(DummySchema)

  config2 = RuntimeConfig(plugin_settings={"my_int": 5})
  res = config2.parse_plugin_settings(DummySchema)
  assert res.my_int == 5


def test_validate_framework_invalid():
  """Auto-generated doc."""
  with pytest.raises(ValueError, match="Unknown framework"):
    RuntimeConfig(source_framework="unknown_framework_123")


def test_parse_cli_key_values():
  """Auto-generated doc."""
  assert parse_cli_key_values(None) == {}
  # test invalid format
  assert parse_cli_key_values(["invalid"]) == {}

  res = parse_cli_key_values(["k1=v1", "k2=true", "k3=FALSE", "k4=1.23", "k5=42", "k6=1e5"])
  assert res == {"k1": "v1", "k2": True, "k3": False, "k4": 1.23, "k5": 42, "k6": 100000.0}

  # Test float parse error
  res2 = parse_cli_key_values(["k7=bad_int"])
  assert res2 == {"k7": "bad_int"}


def test_load_toml_settings_error(tmp_path):
  """Auto-generated doc."""
  # test malformed toml
  toml_file = tmp_path / "pyproject.toml"
  toml_file.write_text("invalid_toml = [")
  settings, path = _load_toml_settings(tmp_path)
  assert settings == {}
  assert path is None

  # test missing tomllib
  with patch("ml_switcheroo.config.tomllib", None):
    settings, path = _load_toml_settings(tmp_path)
    assert settings == {}
    assert path is None
