"""Test suite for the Config Toml module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig, RuntimeConfigOverrides


@pytest.fixture
def toml_file(tmp_path: Path) -> Path:
  """Create a configured `Path` fixture for config TOML tests.

  Args:
      tmp_path: Pytest temporary directory path fixture.

  Returns:
      Configured `toml_file` fixture value (Path).
  """
  fpath: Path = tmp_path / "pyproject.toml"
  content: str = '\n[tool.ml_switcheroo]\nsource_framework = "tensorflow"\ntarget_framework = "mlx"\nstrict_mode = true\n\n[tool.ml_switcheroo.plugin_settings]\nepsilon = 0.005\nuse_gpu = false\ndebug_level = "info"\n'
  fpath.write_text(content, encoding="utf-8")
  return fpath


def test_load_defaults_from_toml(tmp_path: Path, toml_file: Path) -> None:
  """Loads defaults from toml.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      toml_file: Input `toml_file` parameter value (`Path`).
  """
  config: RuntimeConfig = RuntimeConfig.load(search_path=tmp_path)
  assert config.source_framework == "tensorflow"
  assert config.target_framework == "mlx"
  assert config.strict_mode is True
  assert config.plugin_settings["epsilon"] == 0.005
  assert config.plugin_settings["debug_level"] == "info"


def test_cli_overrides_toml(tmp_path: Path, toml_file: Path) -> None:
  """Verify `RuntimeConfig.load` handles CLI overrides TOML as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      toml_file: Input `toml_file` parameter value (`Path`).
  """
  config: RuntimeConfig = RuntimeConfig.load(source="torch", search_path=tmp_path)
  assert config.source_framework == "torch"
  assert config.target_framework == "mlx"
  assert config.strict_mode is True


def test_plugin_settings_merge(tmp_path: Path, toml_file: Path) -> None:
  """Verify `RuntimeConfig.load` handles plugin settings merge as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      toml_file: Input `toml_file` parameter value (`Path`).
  """
  cli_plugins: dict[str, JsonValue] = {"epsilon": 1.0, "new_flag": "yes"}
  config: RuntimeConfig = RuntimeConfig.load(
    RuntimeConfigOverrides(search_path=tmp_path, plugin_settings=cli_plugins),
  )
  assert config.plugin_settings["epsilon"] == 1.0
  assert config.plugin_settings["use_gpu"] is False
  assert config.plugin_settings["new_flag"] == "yes"


def test_hierarchical_search(tmp_path: Path, toml_file: Path) -> None:
  """Verify `RuntimeConfig.load` handles hierarchical search as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      toml_file: Input `toml_file` parameter value (`Path`).
  """
  subdir: Path = tmp_path / "src" / "subdir"
  subdir.mkdir(parents=True)
  config: RuntimeConfig = RuntimeConfig.load(search_path=subdir)
  assert config.source_framework == "tensorflow"


def test_no_toml_fallback(tmp_path: Path) -> None:
  """Verify `RuntimeConfig.load` handles no TOML fallback as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["torch", "jax"]):
    config: RuntimeConfig = RuntimeConfig.load(search_path=tmp_path)
    assert config.source_framework == "torch"
    assert config.target_framework == "jax"
    assert config.strict_mode is False
    assert config.plugin_settings == {}


def test_malformed_toml_is_ignored(tmp_path: Path) -> None:
  """Verify `RuntimeConfig.load` handles malformed TOML is ignored as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  fpath: Path = tmp_path / "pyproject.toml"
  fpath.write_text("[tool.ml_switcheroo\nbad_syntax_here", encoding="utf-8")
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["torch", "jax"]):
    config: RuntimeConfig = RuntimeConfig.load(search_path=tmp_path)
    assert config.source_framework == "torch"
