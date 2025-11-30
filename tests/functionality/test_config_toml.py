"""
Tests for Config Persistence (TOML).

Verifies that:
1. RuntimeConfig.load() picks up [tool.ml_switcheroo] from pyproject.toml.
2. CLI arguments override TOML settings.
3. Plugin settings are merged (CLI wins collisions).
4. File traversal finds toml in parent directories.
"""

import pytest
from ml_switcheroo.config import RuntimeConfig


@pytest.fixture
def toml_file(tmp_path):
  """Creates a dummy pyproject.toml in the temp dir."""
  fpath = tmp_path / "pyproject.toml"
  content = """
[tool.ml_switcheroo]
source_framework = "tensorflow"
target_framework = "mlx"
strict_mode = true

[tool.ml_switcheroo.plugin_settings]
epsilon = 0.005
use_gpu = false
debug_level = "info"
"""
  fpath.write_text(content, encoding="utf-8")
  return fpath


def test_load_defaults_from_toml(tmp_path, toml_file):
  """
  Scenario: User runs CLI without args inside a configured project.
  Expect: Config matches TOML values.
  """
  # Simulate loading from the project root
  config = RuntimeConfig.load(search_path=tmp_path)

  assert config.source_framework == "tensorflow"
  assert config.target_framework == "mlx"
  assert config.strict_mode is True
  assert config.plugin_settings["epsilon"] == 0.005
  assert config.plugin_settings["debug_level"] == "info"


def test_cli_overrides_toml(tmp_path, toml_file):
  """
  Scenario: User provides CLI arg which contradicts TOML.
  Expect: CLI arg takes precedence.
  """
  config = RuntimeConfig.load(
    source="torch",  # Override TOML's tensorflow
    search_path=tmp_path,
  )

  assert config.source_framework == "torch"  # CLI wins
  assert config.target_framework == "mlx"  # TOML fallback
  assert config.strict_mode is True  # TOML fallback


def test_plugin_settings_merge(tmp_path, toml_file):
  """
  Scenario: Plugins configured in TOML, but one value overridden via CLI.
  Expect: Merged dict, CLI value wins collision.
  """
  cli_plugins = {"epsilon": 1.0, "new_flag": "yes"}

  config = RuntimeConfig.load(search_path=tmp_path, plugin_settings=cli_plugins)

  # Collision -> CLI wins
  assert config.plugin_settings["epsilon"] == 1.0

  # Unique to TOML -> Preserved
  assert config.plugin_settings["use_gpu"] is False

  # Unique to CLI -> Added
  assert config.plugin_settings["new_flag"] == "yes"


def test_hierarchical_search(tmp_path, toml_file):
  """
  Scenario: Config is in root, but user runs command from src/subdir.
  Expect: Traversing up finds the root pyproject.toml.
  """
  subdir = tmp_path / "src" / "subdir"
  subdir.mkdir(parents=True)

  config = RuntimeConfig.load(search_path=subdir)

  assert config.source_framework == "tensorflow"


def test_no_toml_fallback(tmp_path):
  """
  Scenario: No pyproject.toml exists.
  Expect: Default hardcoded values (torch -> jax).
  """
  # Ensure no file exists
  config = RuntimeConfig.load(search_path=tmp_path)

  assert config.source_framework == "torch"
  assert config.target_framework == "jax"
  assert config.strict_mode is False
  assert config.plugin_settings == {}


def test_malformed_toml_is_ignored(tmp_path):
  """
  Scenario: pyproject.toml is invalid syntax.
  Expect: Silent ignore, fallback to defaults (robustness).
  """
  fpath = tmp_path / "pyproject.toml"
  fpath.write_text("[tool.ml_switcheroo\nbad_syntax_here", encoding="utf-8")

  config = RuntimeConfig.load(search_path=tmp_path)

  # Should fall back to defaults rather than crashing
  assert config.source_framework == "torch"
