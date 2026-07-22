"""Test suite for the Suggest module."""

import pytest
from unittest.mock import patch
from ml_switcheroo.cli.handlers.suggest import handle_suggest, _inspect_live_object


def test_handle_suggest_single_success(capsys):
  """Handles suggest single successfully."""
  with patch("ml_switcheroo.cli.handlers.suggest._inspect_live_object") as mock_inspect:
    mock_inspect.return_value = {"signature": "()", "docstring": "docs", "kind": "function"}
    assert handle_suggest("foo.bar") == 0


def test_inspect_live_object():
  """Verifies the behavior of inspect live object."""
  meta = _inspect_live_object("os.path.join")
  assert "signature" in meta


def test_inspect_live_object_invalid_path():
  """Verifies the behavior of inspect live object invalid path."""
  with pytest.raises(ImportError):
    _inspect_live_object("os")
