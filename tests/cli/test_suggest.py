"""Test suite for the Suggest module."""

import pytest
from unittest.mock import patch
from ml_switcheroo.cli.handlers.suggest import _inspect_live_object, handle_suggest
import ml_switcheroo.cli.handlers.suggest


def test_inspect_invalid_path():
  """Verifies the behavior of inspect invalid path."""
  with pytest.raises(ImportError, match="Invalid path format: no_dot"):
    _inspect_live_object("no_dot")


def test_handle_suggest_invalid():
  """Handles suggest invalid."""
  assert handle_suggest("no_dot") == 1


def test_suggest_with_module_and_exception(tmp_path):
  """Tests the suggest command covering modules and exceptions."""
  original = getattr(ml_switcheroo.cli.handlers.suggest, "_extract_metadata")

  def mock_extract(obj):
    if getattr(obj, "__name__", "") == "system":
      raise Exception("Mock error")
    return original(obj)

  with patch.object(ml_switcheroo.cli.handlers.suggest, "_extract_metadata", side_effect=mock_extract):
    res = ml_switcheroo.cli.handlers.suggest.handle_suggest("os.*", tmp_path / "subdir", batch_size=2)
  assert res == 0
  assert (tmp_path / "subdir").exists()


def test_extract_metadata_value_error():
  """Test extract_metadata fallback."""
  import _ast

  res = ml_switcheroo.cli.handlers.suggest._extract_metadata(_ast.AST)
  assert res["signature"] == "Unknown Signature"
