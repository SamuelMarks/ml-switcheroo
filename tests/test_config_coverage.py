"""Test suite for the Config Coverage module."""

import types
from pathlib import Path
from typing import Any, Dict, Optional, Union
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from ml_switcheroo.config import (
  RuntimeConfig,
  _import_tomllib,
  _load_toml_settings,
  _resolve_default_source,
  _resolve_default_target,
  parse_cli_key_values,
)


def test_resolve_default_placeholders() -> None:
  """Resolves default placeholders."""
  with patch("ml_switcheroo.config.get_framework_priority_order", return_value=[]):
    assert _resolve_default_source() == "source_placeholder"
    assert _resolve_default_target() == "target_placeholder"
  with patch("ml_switcheroo.config.get_framework_priority_order", return_value=["one"]):
    assert _resolve_default_source() == "one"
    assert _resolve_default_target() == "one"


def test_effective_frameworks() -> None:
  """Verifies the behavior of effective frameworks."""
  config: RuntimeConfig = RuntimeConfig(
    source_framework="torch", target_framework="jax", source_flavour="torch_flavour", target_flavour="jax_flavour"
  )
  assert config.effective_source == "torch_flavour"
  assert config.effective_target == "jax_flavour"
  config2: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  assert config2.effective_source == "torch"
  assert config2.effective_target == "jax"


class DummySchema(BaseModel):
  """Docstring."""

  my_int: int = Field(...)


def test_parse_plugin_settings_error() -> None:
  """Parses plugin settings correctly handling an error."""
  config: RuntimeConfig = RuntimeConfig(plugin_settings={"my_int": "not_an_int"})
  with pytest.raises(ValueError, match="Plugin configuration validation failed"):
    config.parse_plugin_settings(DummySchema)
  config2: RuntimeConfig = RuntimeConfig(plugin_settings={"my_int": 5})
  res: DummySchema = config2.parse_plugin_settings(DummySchema)
  assert res.my_int == 5


def test_validate_framework_invalid() -> None:
  """Validates framework invalid."""
  with pytest.raises(ValueError, match="Unknown framework"):
    RuntimeConfig(source_framework="unknown_framework_123")


def test_parse_cli_key_values() -> None:
  """Parses CLI key values."""
  assert parse_cli_key_values(None) == {}
  assert parse_cli_key_values(["invalid"]) == {}
  res: Dict[str, Union[str, bool, int, float]] = parse_cli_key_values(
    ["k1=v1", "k2=true", "k3=FALSE", "k4=1.23", "k5=42", "k6=1e5"]
  )
  assert res == {"k1": "v1", "k2": True, "k3": False, "k4": 1.23, "k5": 42, "k6": 100000.0}
  res2: Dict[str, Union[str, bool, int, float]] = parse_cli_key_values(["k7=bad_int"])
  assert res2 == {"k7": "bad_int"}


def test_load_toml_settings_error(tmp_path: Path) -> None:
  """Loads toml settings correctly handling an error."""
  toml_file: Path = tmp_path / "pyproject.toml"
  toml_file.write_text("invalid_toml = [")

  settings: Dict[str, Union[str, int, float, bool, list[str]]]
  path: Optional[Path]
  (settings, path) = _load_toml_settings(tmp_path)
  assert settings == {}
  assert path is None
  with patch("ml_switcheroo.config.tomllib", None):
    (settings, path) = _load_toml_settings(tmp_path)
    assert settings == {}
    assert path is None


def test_import_tomllib() -> None:
  """Docstring."""
  # Python 3.11+ mock
  with patch("sys.version_info", (3, 11)):
    with patch("builtins.__import__") as mock_import:
      _import_tomllib()
      assert mock_import.call_args[0][0] == "tomllib"

  # Python < 3.11 but tomli not installed
  with patch("sys.version_info", (3, 9)):
    import builtins

    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
      """Fake import."""
      if name == "tomli":
        raise ImportError("No module named tomli")
      return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
      assert _import_tomllib() is None


def test_runtime_config_load_overrides(tmp_path: Path) -> None:
  """Docstring."""
  toml_file: Path = tmp_path / "pyproject.toml"
  toml_file.write_text("""
[tool.ml_switcheroo]
plugin_paths = ["custom_plugins"]
  """)
  config: RuntimeConfig = RuntimeConfig.load(
    strict_mode=True, enable_graph_optimization=True, enable_sharding=True, search_path=tmp_path
  )
  assert config.strict_mode is True
  assert config.enable_graph_optimization is True
  assert config.enable_sharding is True
  assert len(config.plugin_paths) == 1
  assert config.plugin_paths[0].name == "custom_plugins"


def test_framework_priority_hierarchy_and_missing_priority() -> None:
  """Test framework priority ordering for child flavours and missing ui_priority."""
  from ml_switcheroo.config import get_framework_priority_order

  class ChildAdapter:
    """Mock child adapter."""

    inherits_from = "parent"

  class ParentAdapter:
    """Mock parent adapter with empty inherits_from."""

    inherits_from = None

  class NoPriorityAdapter:
    """Mock adapter without ui_priority."""

    pass

  with patch(
    "ml_switcheroo.frameworks.base.available_frameworks", return_value=["child", "parent", "noprio", "no_adapter"]
  ):
    with patch("ml_switcheroo.frameworks.base.get_adapter") as mock_get:

      def get_adapter_mock(name: str) -> Any:
        """Mock adapter resolver.

        Args:
            name: Framework name.

        Returns:
            Adapter instance or None.
        """
        if name == "child":
          return ChildAdapter()
        elif name == "parent":
          return ParentAdapter()
        elif name == "noprio":
          return NoPriorityAdapter()
        return None

      mock_get.side_effect = get_adapter_mock
      order = get_framework_priority_order()
      assert order == ["no_adapter", "noprio", "parent", "child"]
