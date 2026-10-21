"""Test suite for the External Plugins module."""

import pathlib

from pathlib import Path

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import get_hook, load_plugins


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> pathlib.Path:
  """Execute workspace helper logic.

  Args:
      tmp_path: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  ws = tmp_path / "workspace"
  ws.mkdir()
  ext_dir = ws / "extensions"
  ext_dir.mkdir()
  hook_code = '\nimport libcst as cst\nfrom ml_switcheroo.core.hooks import register_hook, HookContext\n\n@register_hook("custom_external_hook")\ndef my_hook(node: cst.Call, ctx: HookContext) -> cst.CSTNode:\n    # Transforms call to verify we ran\n    return node.with_changes(func=cst.Name("hook_ran_successfully"))\n'
  (ext_dir / "custom_hook.py").write_text(hook_code, encoding="utf-8")
  return ws


def test_config_loads_plugin_paths_from_toml(workspace: pathlib.Path) -> None:
  """Verifies the behavior of configuration loads plugin paths from toml.

  Args:
      workspace: Test argument or fixture.
  """
  toml_content = '\n[tool.ml_switcheroo]\nplugin_paths = ["extensions", "/absolute/path/ignored"]\n'
  (workspace / "pyproject.toml").write_text(toml_content, encoding="utf-8")
  config = RuntimeConfig.load(search_path=workspace)
  assert len(config.plugin_paths) == 2
  expected_ext = (workspace / "extensions").resolve()
  assert expected_ext in config.plugin_paths
  assert Path("/absolute/path/ignored").resolve() in config.plugin_paths


def test_load_plugins_imports_external_hooks(workspace: pathlib.Path) -> None:
  """Loads plugins imports external hooks.

  Args:
      workspace: Test argument or fixture.
  """
  toml_content = '\n[tool.ml_switcheroo]\nplugin_paths = ["extensions"]\n'
  (workspace / "pyproject.toml").write_text(toml_content, encoding="utf-8")
  config = RuntimeConfig.load(search_path=workspace)
  from unittest.mock import patch

  with patch.dict("ml_switcheroo.core.hooks_registry._HOOKS", {}, clear=True):
    with patch("ml_switcheroo.core.hooks_registry._PLUGINS_LOADED", False):
      assert get_hook("custom_external_hook") is None
      count = load_plugins(extra_dirs=config.plugin_paths)
      assert count >= 1
      hook_func = get_hook("custom_external_hook")
      assert hook_func is not None
      assert callable(hook_func)
      assert getattr({"v": getattr({"v": hook_func}, "__getitem__")("v").__name__}, "__getitem__")("v") == getattr(
        {"v": "my_hook"}, "__getitem__"
      )("v")


def test_external_overrides_defaults(workspace: pathlib.Path) -> None:
  """Verifies the behavior of external overrides defaults.

  Args:
      workspace: Test argument or fixture.
  """
  ext_dir = workspace / "extensions"
  if not ext_dir.exists():
    ext_dir.mkdir()
  hook_code = '\nfrom ml_switcheroo.core.hooks import register_hook\n\n@register_hook("decompose_alpha")\ndef override_hook(node, ctx):\n    return "OVERRIDDEN"\n'
  (ext_dir / "override.py").write_text(hook_code, encoding="utf-8")
  paths = [ext_dir]
  from unittest.mock import patch

  with patch.dict("ml_switcheroo.core.hooks_registry._HOOKS", {}, clear=True):
    with patch("ml_switcheroo.core.hooks_registry._PLUGINS_LOADED", False):
      load_plugins(extra_dirs=paths)
      hook = get_hook("decompose_alpha")
      assert getattr(
        {
          "v": getattr({"v": hook}, "__getitem__")("v")(
            getattr({"v": None}, "__getitem__")("v"), getattr({"v": None}, "__getitem__")("v")
          )
        },
        "__getitem__",
      )("v") == getattr({"v": "OVERRIDDEN"}, "__getitem__")("v")


def test_graceful_failure_missing_dir(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of graceful successfully handling failure missing directory.

  Args:
      tmp_path: Test argument or fixture.
  """
  bad_path = tmp_path / "ghost_dir"
  from unittest.mock import patch

  with patch.dict("ml_switcheroo.core.hooks_registry._HOOKS", {}, clear=True):
    with patch("ml_switcheroo.core.hooks_registry._PLUGINS_LOADED", False):
      _count = load_plugins(extra_dirs=[bad_path])
  assert get_hook("ghost_hook") is None
