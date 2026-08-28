"""Test suite for the Config Toml module."""

import pytest
import typing
from pathlib import Path
from unittest.mock import patch
from ml_switcheroo.config import RuntimeConfig


@pytest.fixture
def toml_file(tmp_path: Path) -> Path:
  """Provides a mock toml file for testing."""
  fpath: Path = tmp_path / "pyproject.toml"
  content: str = '\n[tool.ml_switcheroo]\nsource_framework = "tensorflow"\ntarget_framework = "mlx"\nstrict_mode = true\n\n[tool.ml_switcheroo.plugin_settings]\nepsilon = 0.005\nuse_gpu = false\ndebug_level = "info"\n'
  fpath.write_text(content, encoding="utf-8")
  return fpath


def test_load_defaults_from_toml(tmp_path: Path, toml_file: Path) -> None:
  """Loads defaults from toml."""
  config: RuntimeConfig = RuntimeConfig.load(search_path=tmp_path)
  assert config.source_framework == "tensorflow"
  assert config.target_framework == "mlx"
  assert config.strict_mode is True
  assert config.plugin_settings["epsilon"] == 0.005
  assert config.plugin_settings["debug_level"] == "info"


def test_cli_overrides_toml(tmp_path: Path, toml_file: Path) -> None:
  """Verifies the behavior of CLI overrides toml."""
  config: RuntimeConfig = RuntimeConfig.load(source="torch", search_path=tmp_path)
  assert config.source_framework == "torch"
  assert config.target_framework == "mlx"
  assert config.strict_mode is True


def test_plugin_settings_merge(tmp_path: Path, toml_file: Path) -> None:
  """Verifies the behavior of plugin settings merge."""
  cli_plugins: dict[str, typing.Any] = {"epsilon": 1.0, "new_flag": "yes"}
  config: RuntimeConfig = RuntimeConfig.load(search_path=tmp_path, plugin_settings=cli_plugins)
  assert config.plugin_settings["epsilon"] == 1.0
  assert config.plugin_settings["use_gpu"] is False
  assert config.plugin_settings["new_flag"] == "yes"


def test_hierarchical_search(tmp_path: Path, toml_file: Path) -> None:
  """Verifies the behavior of hierarchical search."""
  subdir: Path = tmp_path / "src" / "subdir"
  subdir.mkdir(parents=True)
  config: RuntimeConfig = RuntimeConfig.load(search_path=subdir)
  assert config.source_framework == "tensorflow"


def test_no_toml_fallback(tmp_path: Path) -> None:
  """Verifies the behavior of no toml fallback."""
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["torch", "jax"]):
    config: RuntimeConfig = RuntimeConfig.load(search_path=tmp_path)
    assert config.source_framework == "torch"
    assert config.target_framework == "jax"
    assert config.strict_mode is False
    assert config.plugin_settings == {}


def test_malformed_toml_is_ignored(tmp_path: Path) -> None:
  """Verifies the behavior of malformed toml is ignored."""
  fpath: Path = tmp_path / "pyproject.toml"
  fpath.write_text("[tool.ml_switcheroo\nbad_syntax_here", encoding="utf-8")
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["torch", "jax"]):
    config: RuntimeConfig = RuntimeConfig.load(search_path=tmp_path)
    assert config.source_framework == "torch"
