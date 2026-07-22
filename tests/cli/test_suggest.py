"""Test suite for the Suggest module."""

import pytest
from ml_switcheroo.cli.handlers.suggest import _inspect_live_object, handle_suggest


def test_inspect_invalid_path():
  """Verifies the behavior of inspect invalid path."""
  with pytest.raises(ImportError, match="Invalid path format: no_dot"):
    _inspect_live_object("no_dot")


def test_handle_suggest_invalid():
  """Handles suggest invalid."""
  assert handle_suggest("no_dot") == 1
