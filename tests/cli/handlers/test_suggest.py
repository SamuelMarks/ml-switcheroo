"""Test suite for the Suggest module."""

import pytest
from unittest.mock import patch
from ml_switcheroo.cli.handlers.suggest import handle_suggest, _inspect_live_object


def test_handle_suggest_single_success(capsys):
  """Handles suggest single successfully.

  Args:
      capsys: ...
  """
  with patch("ml_switcheroo.cli.handlers.suggest._inspect_live_object") as mock_inspect:
    mock_inspect.return_value = {"signature": "()", "docstring": "docs", "kind": "function"}
    assert handle_suggest("foo.bar") == 0


def test_handle_suggest_wildcard_success():
  """Tests handle_suggest with wildcard."""
  with patch("importlib.import_module") as mock_import:
    mock_mod = type("MockMod", (), {})
    mock_mod.func1 = lambda: None
    mock_mod.func2 = lambda: None
    mock_mod._private = lambda: None
    mock_mod.submod = type("SubMod", (), {})  # A module, should be skipped
    mock_import.return_value = mock_mod
    assert handle_suggest("my_module.*") == 0


def test_handle_suggest_wildcard_import_error():
  """Tests handle_suggest with wildcard import error."""
  with patch("importlib.import_module", side_effect=ImportError("Failed")):
    assert handle_suggest("my_module.*") == 1


def test_handle_suggest_single_import_error():
  """Tests handle_suggest with single import error."""
  with patch("ml_switcheroo.cli.handlers.suggest._inspect_live_object", side_effect=ImportError("Failed")):
    assert handle_suggest("my_module.func") == 1


def test_handle_suggest_no_targets():
  """Tests handle_suggest with no valid targets."""
  with patch("importlib.import_module") as mock_import:
    mock_mod = type("MockMod", (), {})
    mock_import.return_value = mock_mod
    assert handle_suggest("empty_module.*") == 1


def test_inspect_live_object():
  """Verifies the behavior of inspect live object."""
  meta = _inspect_live_object("os.path.join")
  assert "signature" in meta


def test_inspect_live_object_invalid_path():
  """Verifies the behavior of inspect live object invalid path."""
  with pytest.raises(ImportError):
    _inspect_live_object("os")
