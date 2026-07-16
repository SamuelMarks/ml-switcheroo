"""Test module."""

import pytest
from ml_switcheroo.cli.handlers.suggest import _inspect_live_object, handle_suggest


def test_inspect_invalid_path():
  """Test function."""
  with pytest.raises(ImportError, match="Invalid path format: no_dot"):
    _inspect_live_object("no_dot")


def test_handle_suggest_invalid():
  """Test function."""
  assert handle_suggest("no_dot") == 1
