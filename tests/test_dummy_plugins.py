"""Test suite for the Dummy Plugins module."""

import importlib
import io
from unittest.mock import patch

import ml_switcheroo.plugins


def test_dummy_plugins_init() -> None:
  """Verifies the behavior of dummy plugins initialization."""
  # Capture stdout to assert the printed warning
  captured_output = io.StringIO()

  with patch("sys.stdout", new=captured_output):
    with patch("importlib.import_module", side_effect=Exception("mocked err")):
      importlib.reload(ml_switcheroo.plugins)

  output_text = captured_output.getvalue()

  # Assert that the fallback behavior logs the warning correctly
  assert "Failed to auto-load plugin" in output_text
  assert "mocked err" in output_text
