"""Test suite for the Hooks Loader module."""

import pytest
from unittest.mock import patch
from ml_switcheroo.core.hooks import load_plugins, get_hook, _HOOKS


@pytest.fixture
def mock_plugin_dir(tmp_path):
  """Provides a mock plugin directory for testing."""
  plugin_dir = tmp_path / "custom_plugins"
  plugin_dir.mkdir()
  plugin_file = plugin_dir / "my_plugin.py"
  plugin_file.write_text(
    "from ml_switcheroo.core.hooks import register_hook\n@register_hook('custom_trigger')\ndef my_hook(node, ctx):\n  return node\n"
  )
  return plugin_dir


def test_load_plugins_default():
  """Loads plugins default."""
  with patch.dict("ml_switcheroo.core.hooks._HOOKS", {}, clear=True):
    with patch("ml_switcheroo.core.hooks_registry._PLUGINS_LOADED", False):
      count = load_plugins()
      assert count > 0
      assert get_hook("inject_prng") is not None


def test_load_plugins_from_custom_dir(mock_plugin_dir):
  """Loads plugins from custom directory."""
  with patch.dict("ml_switcheroo.core.hooks_registry._HOOKS", {}, clear=True):
    with patch("ml_switcheroo.core.hooks_registry._PLUGINS_LOADED", False):
      count = load_plugins(extra_dirs=[mock_plugin_dir])
      assert count > 0
      assert "custom_trigger" in _HOOKS
