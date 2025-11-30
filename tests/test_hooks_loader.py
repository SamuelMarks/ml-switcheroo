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
  Verify argument-less call attempts to resolve swithcheroo/plugins.
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
  sys.modules state without explicit install, we manually set the plugins_dir
  to the real source location using pathlib relative to this test file.
  test is in: tests/core/test_hooks_loader.py
  src is in:  src/ml_switcheroo/plugins
  """
  clear_hooks()

  # Resolve real source path
  # tests/core -> tests -> root -> src -> ml_switcheroo -> plugins
  root_dir = Path(__file__).resolve().parent.parent.parent
  real_plugins_dir = root_dir / "src" / "ml_switcheroo" / "plugins"

  if not real_plugins_dir.exists():
    pytest.skip("Skipping integration test: 'src/ml_switcheroo/plugins' not found on disk.")

  # We patch the default resolution inside load_plugins to use this explicit path
  # OR we just call load_plugins first? No, we checking LAZY load.

  # The `load_plugins()` call inside `get_hook` takes no args, meaning it calculates path via `__file__`.
  # `src/ml_switcheroo/core/hooks.py` uses `Path(__file__).parent.parent / "plugins"`.
  # This calculation IS correct if the file exists on disk.

  # Why did it fail? importlib might fail if "ml_switcheroo" isn't a top-level package in sys.path.
  # We ensure src is in python path.
  sys.path.insert(0, str(root_dir / "src"))

  try:
    # Verify decompose_alpha (known real hook)
    hook = get_hook("decompose_alpha")

    # If still None, it means the import failed, possibly due to module name resolution.
    # But importlib fallback handles file spec loading too.

    assert hook is not None, "Failed to auto-discover standard decompositions"
  finally:
    # Cleanup path
    sys.path.pop(0)
