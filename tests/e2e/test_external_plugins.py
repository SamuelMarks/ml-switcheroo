"""Test suite for the External Plugins module."""

import pathlib
import types
from pathlib import Path

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext, clear_hooks, get_hook, load_plugins


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> pathlib.Path:
  """Create a configured `pathlib.Path` fixture for external plugins tests.

  Args:
      tmp_path: Pytest temporary directory path fixture.

  Returns:
      Configured `workspace` fixture value (pathlib.Path).
  """
  ws = tmp_path / "workspace"
  ws.mkdir()
  ext_dir = ws / "extensions"
  ext_dir.mkdir()
  hook_code = '\nimport libcst as cst\nfrom ml_switcheroo.core.hooks import register_hook, HookContext\n\n@register_hook("custom_external_hook")\ndef my_hook(node: cst.Call, ctx: HookContext) -> cst.CSTNode:\n    # Transforms call to verify we ran\n    return node.with_changes(func=cst.Name("hook_ran_successfully"))\n'
  (ext_dir / "custom_hook.py").write_text(hook_code, encoding="utf-8")
  return ws


def test_config_loads_plugin_paths_from_toml(workspace: pathlib.Path) -> None:
  """Verify `RuntimeConfig.load` handles config loads plugin paths from TOML as expected.

  Args:
      workspace: Input `workspace` parameter value (`pathlib.Path`).
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
      workspace: Input `workspace` parameter value (`pathlib.Path`).
  """
  toml_content = '\n[tool.ml_switcheroo]\nplugin_paths = ["extensions"]\n'
  (workspace / "pyproject.toml").write_text(toml_content, encoding="utf-8")
  config = RuntimeConfig.load(search_path=workspace)

  clear_hooks(mark_loaded=False)
  assert get_hook("custom_external_hook") is None
  count = load_plugins(extra_dirs=config.plugin_paths)
  assert count >= 1
  hook_func = get_hook("custom_external_hook")
  assert hook_func is not None
  assert callable(hook_func)
  assert isinstance(hook_func, types.FunctionType)
  assert hook_func.__name__ == "my_hook"


def test_external_overrides_defaults(workspace: pathlib.Path) -> None:
  """Verify `clear_hooks` handles external overrides defaults as expected.

  Args:
      workspace: Input `workspace` parameter value (`pathlib.Path`).
  """
  ext_dir = workspace / "extensions"
  if not ext_dir.exists():
    ext_dir.mkdir()
  hook_code = '\nimport libcst as cst\nfrom ml_switcheroo.core.hooks import HookContext, register_hook\n\n@register_hook("decompose_alpha")\ndef override_hook(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode:\n    return cst.Name("OVERRIDDEN")\n'
  (ext_dir / "override.py").write_text(hook_code, encoding="utf-8")
  paths = [ext_dir]

  clear_hooks(mark_loaded=False)
  load_plugins(extra_dirs=paths)
  hook = get_hook("decompose_alpha")
  assert hook is not None
  ctx = HookContext(semantics=None, config=RuntimeConfig())
  res = hook(cst.Name("x"), ctx)
  assert isinstance(res, cst.Name)
  assert res.value == "OVERRIDDEN"


def test_graceful_failure_missing_dir(tmp_path: pathlib.Path) -> None:
  """Verify `clear_hooks` handles graceful failure missing dir as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  bad_path = tmp_path / "ghost_dir"

  clear_hooks(mark_loaded=False)
  _count = load_plugins(extra_dirs=[bad_path])
  assert get_hook("ghost_hook") is None
