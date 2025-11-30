"""
Tests for Configuration, Path Resolution, and CLI Parameter Passing.
Now verifying Pydantic integration and clean-slate architecture.
"""

import pytest
from pathlib import Path
from pydantic import BaseModel, ValidationError

from ml_switcheroo.semantics.manager import resolve_semantics_dir, SemanticsManager
from ml_switcheroo.config import RuntimeConfig, parse_cli_key_values
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.enums import SupportedEngine


def test_resolve_semantics_dir_returns_path():
  """Verify semantics path resolution."""
  path = resolve_semantics_dir()
  assert isinstance(path, Path)
  # The path should exist in the source tree/install (even if empty)
  assert path.exists()


def test_semantics_manager_uses_resolved_path(tmp_path, monkeypatch):
  """
  Verify loading mechanic respects the resolved path.

  Since we removed hardcoded defaults (Specification-Driven Architecture),
  we must inject a file to prove the manager reads from the resolved location.
  """
  # 1. Setup a dummy specification in the temp path
  # valid JSON content required to avoid decode errors
  (tmp_path / "k_array_api.json").write_text('{"mock_op": {"variants": {}}}')

  # 2. Mock resolution to point to tmp_path
  def mock_resolve():
    return tmp_path

  monkeypatch.setattr("ml_switcheroo.semantics.manager.resolve_semantics_dir", mock_resolve)

  mgr = SemanticsManager()

  # 3. Verify it loaded the file we put there
  assert "mock_op" in mgr.data

  # 4. Verify no defaults existed (clean slate architecture)
  assert "test_alpha_add" not in mgr.data


def test_cli_parsing_creates_valid_pydantic_model():
  """Verify flow from CLI strings -> Dict -> Pydantic Model."""
  user_args = ["epsilon=1e-5", "optimize=true"]
  parsed = parse_cli_key_values(user_args)

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", plugin_settings=parsed)

  # Assert Attributes match
  assert cfg.source_framework == SupportedEngine.TORCH
  assert cfg.plugin_settings["epsilon"] == 1e-5
  assert cfg.plugin_settings["optimize"] is True


def test_invalid_framework_raises_validation_error():
  """
  Verify that providing an invalid framework string raises a Pydantic Validation Error.
  """
  with pytest.raises(ValidationError) as excinfo:
    RuntimeConfig(source_framework="py-torch")

  assert "source_framework" in str(excinfo.value)


def test_load_promotes_strings_to_enums():
  """
  Verify RuntimeConfig.load correctly coerces string inputs to Enums.
  """
  cfg = RuntimeConfig.load(source="tensorflow", target="mlx")

  assert cfg.source_framework == SupportedEngine.TENSORFLOW
  assert isinstance(cfg.source_framework, SupportedEngine)
  assert cfg.target_framework == SupportedEngine.MLX


def test_hook_context_validates_plugin_schema():
  """
  Verify that plugins can enforce their own schemas using Pydantic.
  """
  # 1. Setup Global Config (Unstructured)
  raw_settings = {"epsilon": 0.001, "ignored_key": "junk"}
  global_cfg = RuntimeConfig(plugin_settings=raw_settings)

  ctx = HookContext(None, global_cfg)

  # 2. Define Plugin-Specific Schema
  class MyPluginConfig(BaseModel):
    epsilon: float = 1e-4  # Default

  # 3. Validate
  validated = ctx.validate_settings(MyPluginConfig)

  assert validated.epsilon == 0.001  # Overridden by raw_settings
  assert not hasattr(validated, "ignored_key")  # Extra keys stripped


def test_hook_context_validation_failure():
  """Verify validation errors are raised if CLI provided bad types."""
  # User passed a string where float expected
  raw_settings = {"epsilon": "not_a_number"}
  global_cfg = RuntimeConfig(plugin_settings=raw_settings)

  ctx = HookContext(None, global_cfg)

  class MyPluginConfig(BaseModel):
    epsilon: float

  with pytest.raises(ValidationError):
    ctx.validate_settings(MyPluginConfig)
