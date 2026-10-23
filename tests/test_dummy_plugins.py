"""Test suite for the Dummy Plugins module."""

import importlib
from unittest.mock import patch

import ml_switcheroo.plugins


def test_dummy_plugins_init() -> None:
  """Verifies the behavior of dummy plugins initialization logging on error."""
  import sys

  sys.modules.pop("ml_switcheroo.plugins.in_top_k_plugin", None)
  with patch("logging.Logger.warning") as mock_warn:
    with patch("importlib.import_module", side_effect=ImportError("mocked err")):
      importlib.reload(ml_switcheroo.plugins)

  assert mock_warn.called
  warning_msg = str(mock_warn.call_args[0][0])
  assert "Failed to load plugin" in warning_msg
  assert "mocked err" in warning_msg
