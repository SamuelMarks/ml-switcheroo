"""Test suite for the Dummy Plugins module."""

from unittest.mock import patch

from ml_switcheroo.core.hooks import clear_hooks, load_plugins


def test_dummy_plugins_init() -> None:
  """Verify `pop` emits `Failed to load plugin` for dummy plugins init."""
  import sys

  sys.modules.pop("ml_switcheroo.plugins.in_top_k_plugin", None)
  with (
    patch("logging.Logger.warning") as mock_warn,
    patch("importlib.import_module", side_effect=ImportError("mocked err")),
  ):
    clear_hooks(mark_loaded=False)
    load_plugins()

  assert mock_warn.called
  warning_msg = str(mock_warn.call_args[0][0])
  assert "Failed to load plugin" in warning_msg
  assert "mocked err" in warning_msg
