"""
Tests for Dynamic Plugin Loading.
"""

import sys
import pytest
from pathlib import Path
from ml_switcheroo.core.hooks import load_plugins, get_hook, clear_hooks, _HOOKS


@pytest.fixture
def mock_plugin_dir(tmp_path):
  """
  Creates a temporary directory with a valid python module containing
  a ml-switcheroo hook.
  """
  plugin_dir = tmp_path / "custom_plugins"
  plugin_dir.mkdir()

  # Create valid plugin
  plugin_file = plugin_dir / "my_plugin.py"
  plugin_file.write_text(""" 
from ml_switcheroo.core.hooks import register_hook

@register_hook("dynamic_test_trigger") 
def my_hook(node, ctx): 
    return node
""")

  # Create __init__ (should be ignored by naive recursor or treated as package)
  (plugin_dir / "__init__.py").touch()

  return plugin_dir


def test_load_plugins_from_custom_dir(mock_plugin_dir):
  """Verify loading from a specific path works."""
  clear_hooks()

  count = load_plugins(mock_plugin_dir)

  assert count == 1
  assert "dynamic_test_trigger" in _HOOKS

  hook = get_hook("dynamic_test_trigger")
  assert hook is not None


def test_load_plugins_default_location(monkeypatch):
  """
  Verify argument-less call attempts to resolve ml_switcheroo/plugins.

  We mock Path.exists to fail safely since we can't easily rely on
  installed package state in unit tests without side effects.
  """
  clear_hooks()

  # Mocking exists to avoid crashing if dir is missing in test env
  # but actual test is ensuring it *tries* to find it.

  def mock_exists(_self):
    # For the test, we pretend the default dir is missing so it returns 0 safely
    # Testing logic path, not filesystem.
    return False

  monkeypatch.setattr(Path, "exists", mock_exists)

  count = load_plugins()
  # Should not crash
  assert count == 0


def test_lazy_loading_in_get_hook(monkeypatch):
  """
  Verify get_hook triggers load_plugins if not loaded.

  Since we cannot guarantee 'ml_switcheroo.plugins' is resolvable in the test runner's
  sys.modules state without explicit install (or if it was cached by previous tests),
  we manually cleanup sys.modules to force a fresh import of the plugins package.
  """
  clear_hooks()

  # Resolve real source path
  # tests/core -> tests -> root -> src -> ml_switcheroo -> plugins
  root_dir = Path(__file__).resolve().parent.parent
  real_plugins_dir = root_dir / "src" / "ml_switcheroo" / "plugins"

  if not real_plugins_dir.exists():
    pytest.skip("Skipping integration test: 'src/ml_switcheroo/plugins' not found on disk.")

  # Force unload of plugins to verify auto-discovery actually works
  # If we don't do this, 'import ml_switcheroo.plugins' inside load_plugins() is a no-op
  # if previously imported, skipping the __init__.py scanning logic.
  mods_to_remove = [m for m in sys.modules if m.startswith("ml_switcheroo.plugins")]
  for m in mods_to_remove:
    del sys.modules[m]

  # We ensure src is in python path to allow importlib to work on "ml_switcheroo.plugins"
  sys.path.insert(0, str(root_dir / "src"))

  try:
    # Verify a known real hook exists in the codebase.
    # We use 'batch_norm_unwrap' (from plugins/batch_norm.py) as it is a core feature.
    # Calling get_hook should trigger load_plugins() which scans and imports.
    hook = get_hook("batch_norm_unwrap")

    assert hook is not None, "Failed to auto-discover standard plugins (batch_norm)"
  finally:
    # Cleanup path
    if str(root_dir / "src") in sys.path:
      sys.path.remove(str(root_dir / "src"))
