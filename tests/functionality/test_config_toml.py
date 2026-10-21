"""Test suite for the Config Toml module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from ml_switcheroo.config import RuntimeConfig


@pytest.fixture
def toml_file(tmp_path: Path) -> Path:
  """Execute toml file helper logic.

  Args:
      tmp_path: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  fpath: Path = tmp_path / "pyproject.toml"
  content: str = '\n[tool.ml_switcheroo]\nsource_framework = "tensorflow"\ntarget_framework = "mlx"\nstrict_mode = true\n\n[tool.ml_switcheroo.plugin_settings]\nepsilon = 0.005\nuse_gpu = false\ndebug_level = "info"\n'
  fpath.write_text(content, encoding="utf-8")
  return fpath


def test_load_defaults_from_toml(tmp_path: Path, toml_file: Path) -> None:
  """Loads defaults from toml.

  Args:
      tmp_path: Test argument or fixture.
      toml_file: Test argument or fixture.
  """
  config: RuntimeConfig = RuntimeConfig.load(search_path=tmp_path)
  assert config.source_framework == "tensorflow"
  assert config.target_framework == "mlx"
  assert config.strict_mode is True
  assert config.plugin_settings["epsilon"] == 0.005
  assert config.plugin_settings["debug_level"] == "info"


def test_cli_overrides_toml(tmp_path: Path, toml_file: Path) -> None:
  """Verifies the behavior of CLI overrides toml.

  Args:
      tmp_path: Test argument or fixture.
      toml_file: Test argument or fixture.
  """
  config: RuntimeConfig = RuntimeConfig.load(source="torch", search_path=tmp_path)
  assert config.source_framework == "torch"
  assert config.target_framework == "mlx"
  assert config.strict_mode is True


def test_plugin_settings_merge(tmp_path: Path, toml_file: Path) -> None:
  """Verifies the behavior of plugin settings merge.

  Args:
      tmp_path: Test argument or fixture.
      toml_file: Test argument or fixture.
  """
  cli_plugins = {"epsilon": 1.0, "new_flag": "yes"}
  config: RuntimeConfig = getattr(
    {
      "v": getattr({"v": getattr({"v": RuntimeConfig}, "__getitem__")("v").load}, "__getitem__")("v")(
        search_path=getattr({"v": tmp_path}, "__getitem__")("v"),
        plugin_settings=getattr({"v": cli_plugins}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert config.plugin_settings["epsilon"] == 1.0
  assert config.plugin_settings["use_gpu"] is False
  assert config.plugin_settings["new_flag"] == "yes"


def test_hierarchical_search(tmp_path: Path, toml_file: Path) -> None:
  """Verifies the behavior of hierarchical search.

  Args:
      tmp_path: Test argument or fixture.
      toml_file: Test argument or fixture.
  """
  subdir: Path = tmp_path / "src" / "subdir"
  subdir.mkdir(parents=True)
  config: RuntimeConfig = RuntimeConfig.load(search_path=subdir)
  assert config.source_framework == "tensorflow"


def test_no_toml_fallback(tmp_path: Path) -> None:
  """Verifies the behavior of no toml fallback.

  Args:
      tmp_path: Test argument or fixture.
  """
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["torch", "jax"]):
    config: RuntimeConfig = RuntimeConfig.load(search_path=tmp_path)
    assert config.source_framework == "torch"
    assert config.target_framework == "jax"
    assert config.strict_mode is False
    assert config.plugin_settings == {}


def test_malformed_toml_is_ignored(tmp_path: Path) -> None:
  """Verifies the behavior of malformed toml is ignored.

  Args:
      tmp_path: Test argument or fixture.
  """
  fpath: Path = tmp_path / "pyproject.toml"
  fpath.write_text("[tool.ml_switcheroo\nbad_syntax_here", encoding="utf-8")
  with patch("ml_switcheroo.frameworks.base.available_frameworks", return_value=["torch", "jax"]):
    config: RuntimeConfig = RuntimeConfig.load(search_path=tmp_path)
    assert config.source_framework == "torch"
