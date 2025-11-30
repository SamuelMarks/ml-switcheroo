"""
Integration Tests for External Plugin Loading.

Verifies that:
1. `RuntimeConfig.load` parses `plugin_paths` from `pyproject.toml`.
2. Paths are reliably resolved relative to the TOML file.
3. `load_plugins` correctly imports modules from these external paths.
4. Custom hooks are registered and callable via the registry.
"""

import pytest
from pathlib import Path
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import load_plugins, get_hook, clear_hooks


@pytest.fixture
def workspace(tmp_path):
  """
  Creates a mock project structure:
  /workspace
    pyproject.toml
    /extensions
      custom_hook.py
  """
  ws = tmp_path / "workspace"
  ws.mkdir()

  # Create extension directory
  ext_dir = ws / "extensions"
  ext_dir.mkdir()

  # Create a custom plugin file
  hook_code = """
import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext

@register_hook("custom_external_hook")
def my_hook(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
    # Transforms call to verify we ran
    return node.with_changes(func=cst.Name("hook_ran_successfully"))
"""
  (ext_dir / "custom_hook.py").write_text(hook_code, encoding="utf-8")

  return ws


def test_config_loads_plugin_paths_from_toml(workspace):
  """
  Verify RuntimeConfig parses [tool.ml_switcheroo] plugin_paths.
  """
  toml_content = """
[tool.ml_switcheroo]
plugin_paths = ["extensions", "/absolute/path/ignored"]
"""
  (workspace / "pyproject.toml").write_text(toml_content, encoding="utf-8")

  # Load config from workspace root
  config = RuntimeConfig.load(search_path=workspace)

  assert len(config.plugin_paths) == 2

  # Verify relative path resolution
  expected_ext = (workspace / "extensions").resolve()
  assert expected_ext in config.plugin_paths

  # Verify absolute path parsing (simple check)
  assert Path("/absolute/path/ignored").resolve() in config.plugin_paths


def test_load_plugins_imports_external_hooks(workspace):
  """
  Verify load_plugins actually executes the code in external directories.
  """
  # 1. Setup Config
  toml_content = """
[tool.ml_switcheroo]
plugin_paths = ["extensions"]
"""
  (workspace / "pyproject.toml").write_text(toml_content, encoding="utf-8")

  config = RuntimeConfig.load(search_path=workspace)

  # 2. Reset Registry
  clear_hooks()
  assert get_hook("custom_external_hook") is None

  # 3. Trigger Load
  # Note: In real app, ASTEngine would call load_plugins with config.plugin_paths
  count = load_plugins(extra_dirs=config.plugin_paths)

  # 4. Verify
  # Count should include at least custom_hook.py (and defaults if found)
  assert count >= 1

  # Check Registry for the specific hook defined in custom_hook.py
  hook_func = get_hook("custom_external_hook")
  assert hook_func is not None
  assert callable(hook_func)

  # Check name to ensure it's not a mock
  assert hook_func.__name__ == "my_hook"


def test_external_overrides_defaults(workspace):
  """
  Verify external plugins can overwrite hooks with the same name.
  """
  # Create a plugin that overwrites 'decompose_alpha' (a standard plugin)
  ext_dir = workspace / "extensions"
  if not ext_dir.exists():
    ext_dir.mkdir()

  hook_code = """
from ml_switcheroo.core.hooks import register_hook

@register_hook("decompose_alpha")
def override_hook(node, ctx):
    return "OVERRIDDEN" 
"""
  (ext_dir / "override.py").write_text(hook_code, encoding="utf-8")

  # Setup Paths
  paths = [ext_dir]

  # Reset
  clear_hooks()

  # Load
  load_plugins(extra_dirs=paths)

  # Verify
  hook = get_hook("decompose_alpha")
  # Execute dummy args to check return value
  assert hook(None, None) == "OVERRIDDEN"


def test_graceful_failure_missing_dir(tmp_path):
  """
  Verify system doesn't crash if config points to non-existent dir.
  """
  # Path that doesn't exist
  bad_path = tmp_path / "ghost_dir"

  # Should run without exception and return 0 (or count of defaults)
  clear_hooks()
  _count = load_plugins(extra_dirs=[bad_path])

  # It might load defaults, but definitely shouldn't crash on bad_path
  # Try getting a hook that doesn't exist
  assert get_hook("ghost_hook") is None
