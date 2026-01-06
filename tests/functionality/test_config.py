"""
Tests for Configuration, Path Resolution, and Defaults Logic.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from ml_switcheroo.config import (
  RuntimeConfig,
  parse_cli_key_values,
  _resolve_default_source,
  _resolve_default_target,
)

# --- 1. Dynamic Defaults Logic Tests ---


def test_defaults_priority_sort():
  """
  Scenario: Two frameworks registered.
  'beta' has high priority (1), 'alpha' has low priority (100).
  Expectation: Source='beta' (1), Target='alpha' (100).
  """
  adapter_alpha = MagicMock()
  adapter_alpha.ui_priority = 100
  adapter_alpha.inherits_from = None

  adapter_beta = MagicMock()
  adapter_beta.ui_priority = 1
  adapter_beta.inherits_from = None

  # Mock registry lookups
  def get_mock_adapter(name):
    if name == "alpha":
      return adapter_alpha
    if name == "beta":
      return adapter_beta
    return None

  # FIX: Patch directly in frameworks.base where logic comes from
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["alpha", "beta"]):
    with patch("ml_switcheroo.frameworks.base.get_adapter", side_effect=get_mock_adapter):
      src = _resolve_default_source()
      tgt = _resolve_default_target()

      # Beta should sort first due to priority 1
      assert src == "beta"
      assert tgt == "alpha"


def test_defaults_no_priority_defined():
  """
  Scenario: Adapters have no ui_priority (defaults to 999).
  Expectation: Sorts alphabetical.
  """
  adapter_a = MagicMock()
  del adapter_a.ui_priority  # Ensure attribute missing

  adapter_b = MagicMock()
  del adapter_b.ui_priority

  def get_mock_adapter(name):
    if name == "a":
      return adapter_a
    if name == "b":
      return adapter_b
    return None

  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["b", "a"]):
    with patch("ml_switcheroo.frameworks.base.get_adapter", side_effect=get_mock_adapter):
      src = _resolve_default_source()
      # Alphabetical fallback: A comes before B
      assert src == "a"
      assert _resolve_default_target() == "b"


def test_defaults_single_framework():
  """
  Scenario: Registry has ['jax'].
  Expectation: Source='jax', Target='jax'.
  """
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["jax"]):
    with patch("ml_switcheroo.frameworks.base.get_adapter", return_value=MagicMock(ui_priority=10)):
      src = _resolve_default_source()
      tgt = _resolve_default_target()
      assert src == "jax"
      assert tgt == "jax"


def test_defaults_empty_registry():
  """
  Scenario: Registry empty.
  Expectation: Placeholders.
  """
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=[]):
    src = _resolve_default_source()
    tgt = _resolve_default_target()
    assert src == "source_placeholder"
    assert tgt == "target_placeholder"


# --- 2. RuntimeConfig Loading (TOML + CLI) Tests ---


@pytest.fixture
def toml_env(tmp_path):
  """Creates a dummy pyproject.toml."""
  fpath = tmp_path / "pyproject.toml"
  content = """ 
[tool.ml_switcheroo] 
source_framework = "tensorflow" 
target_framework = "mlx" 
strict_mode = true

[tool.ml_switcheroo.plugin_settings] 
epsilon = 0.005
use_gpu = false
"""
  fpath.write_text(content, encoding="utf-8")
  return tmp_path


def test_load_defaults_from_toml(toml_env):
  """Run without CLI args inside a configured root."""
  # We must patch available_frameworks because validation runs during init
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["tensorflow", "mlx"]):
    config = RuntimeConfig.load(search_path=toml_env)

  assert config.source_framework == "tensorflow"
  assert config.target_framework == "mlx"
  assert config.strict_mode is True
  assert config.plugin_settings["epsilon"] == 0.005


def test_cli_overrides_toml(toml_env):
  """Run with CLI args overriding TOML."""
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["torch", "mlx"]):
    config = RuntimeConfig.load(
      source="torch",  # Override 'tensorflow'
      strict_mode=False,  # Override 'true'
      search_path=toml_env,
    )

  assert config.source_framework == "torch"
  assert config.target_framework == "mlx"  # Fallback to TOML
  assert config.strict_mode is False


def test_recursive_toml_search(toml_env):
  """Ensure it finds TOML in parent dir."""
  subdir = toml_env / "src" / "subdir"
  subdir.mkdir(parents=True)

  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["tensorflow", "mlx"]):
    config = RuntimeConfig.load(search_path=subdir)

  assert config.source_framework == "tensorflow"


def test_no_toml_library_fallback(toml_env):
  """
  Verify behavior if tomllib is missing (simulate Python < 3.11 without tomli).
  """
  with patch("ml_switcheroo.config.tomllib", None):
    # Explicit mock to force default resolution usage
    mock_adapter = MagicMock(ui_priority=1)
    with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["torch", "jax"]):
      with patch("ml_switcheroo.frameworks.base.get_adapter", return_value=mock_adapter):
        config = RuntimeConfig.load(search_path=toml_env)

  # Should ignore the file and use defaults
  # Since mocked priority is equal (reused mock), alphabetical sort applies: jax, torch
  assert config.source_framework == "jax"
  assert config.target_framework == "torch"


# --- 3. Validation & Parsing Tests ---


def test_invalid_framework_raises_validation_error():
  """Verify validation against registry."""
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["torch"]):
    with pytest.raises(ValidationError) as excinfo:
      RuntimeConfig(source_framework="ghost_fw")
    assert "Unknown framework" in str(excinfo.value)


def test_cli_key_value_parsing():
  """Verify parsing logic for plugin_settings strings."""
  args = ["val=1", "f=1.5", "b=true", "s=hello", "bad_arg"]
  parsed = parse_cli_key_values(args)

  assert parsed["val"] == 1
  assert parsed["f"] == 1.5
  assert parsed["b"] is True
  assert parsed["s"] == "hello"
  assert "bad_arg" not in parsed


def test_effective_flavours():
  """Verify flavour resolution logic."""
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["torch", "jax"]):
    cfg = RuntimeConfig(source_framework="torch", target_framework="jax", target_flavour="flax_nnx")
    assert cfg.effective_source == "torch"
    assert cfg.effective_target == "flax_nnx"
