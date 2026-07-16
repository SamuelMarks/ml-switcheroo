"""Tests for dynamic plugin loader."""

import pytest
from unittest.mock import patch
from ml_switcheroo.core.hooks import load_plugins, get_hook, _HOOKS


# We test with a temporary directory acting as an external plugin folder
@pytest.fixture
def mock_plugin_dir(tmp_path):
  """Create a temporary plugin file."""
  plugin_dir = tmp_path / "custom_plugins"
  plugin_dir.mkdir()

  plugin_file = plugin_dir / "my_plugin.py"
  plugin_file.write_text(
    "from ml_switcheroo.core.hooks import register_hook\n"
    "@register_hook('custom_trigger')\n"
    "def my_hook(node, ctx):\n"
    "  return node\n"
  )
  return plugin_dir


def test_load_plugins_default():
  """Verify standard plugin loading."""
  # We clear the state to ensure a clean load
  with patch.dict("ml_switcheroo.core.hooks._HOOKS", {}, clear=True):
    with patch("ml_switcheroo.core.hooks_registry._PLUGINS_LOADED", False):
      count = load_plugins()
      assert count > 0
      # Verify a known core plugin is loaded
      assert get_hook("inject_prng") is not None


def test_load_plugins_from_custom_dir(mock_plugin_dir):
  """Verify loading from a specific path works."""
  with patch.dict("ml_switcheroo.core.hooks_registry._HOOKS", {}, clear=True):
    with patch("ml_switcheroo.core.hooks_registry._PLUGINS_LOADED", False):
      count = load_plugins(extra_dirs=[mock_plugin_dir])
      assert count > 0
      # Check that the external plugin was loaded
      assert "custom_trigger" in _HOOKS
