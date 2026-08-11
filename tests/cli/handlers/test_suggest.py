"""Test suite for the Suggest module."""

import pytest
from unittest.mock import patch
import ml_switcheroo.cli.handlers.suggest as suggest_module
from ml_switcheroo.cli.handlers.suggest import (
  handle_suggest,
  _inspect_live_object,
  _extract_metadata,
  _build_header,
  _build_footer,
  _build_target_block,
)


def test_handle_suggest_single_success(capsys):
  """Handles suggest single successfully."""
  with patch("ml_switcheroo.cli.handlers.suggest._inspect_live_object") as mock_inspect:
    mock_inspect.return_value = {"signature": "()", "docstring": "docs", "kind": "function"}
    assert handle_suggest("foo.bar") == 0
    captured = capsys.readouterr()
    assert "foo.bar" in captured.out


def test_handle_suggest_wildcard_success(capsys):
  """Tests handle_suggest with wildcard."""
  with patch("importlib.import_module") as mock_import:
    mock_mod = type("MockMod", (), {})
    mock_mod.func1 = lambda: None
    mock_mod.func2 = lambda: None
    mock_mod._private = lambda: None
    import types

    mock_mod.submod = types.ModuleType("submod")
    mock_mod.submod = type("SubMod", (), {})
    mock_import.return_value = mock_mod
    assert handle_suggest("my_module.*") == 0
    captured = capsys.readouterr()
    assert "my_module.func1" in captured.out
    assert "my_module.func2" in captured.out


def test_handle_suggest_wildcard_with_outdir(tmp_path):
  """Tests handle_suggest with wildcard and out_dir."""
  with patch("importlib.import_module") as mock_import:
    mock_mod = type("MockMod", (), {})
    mock_mod.func1 = lambda: None
    mock_mod.func2 = lambda: None
    mock_import.return_value = mock_mod
    assert handle_suggest("my_module.*", out_dir=tmp_path / "out", batch_size=1) == 0
    files = list((tmp_path / "out").glob("*.md"))
    assert len(files) == 2


def test_handle_suggest_wildcard_import_error():
  """Tests handle_suggest with wildcard import error."""
  with patch("importlib.import_module", side_effect=ImportError("Failed")):
    assert handle_suggest("my_module.*") == 1


def test_handle_suggest_single_import_error():
  """Tests handle_suggest with single import error."""
  with patch("ml_switcheroo.cli.handlers.suggest._inspect_live_object", side_effect=ImportError("Failed")):
    assert handle_suggest("my_module.func") == 1


def test_handle_suggest_single_attribute_error():
  """Tests handle_suggest with single attribute error."""
  with patch("ml_switcheroo.cli.handlers.suggest._inspect_live_object", side_effect=AttributeError("Failed")):
    assert handle_suggest("my_module.func") == 1


def test_handle_suggest_no_targets():
  """Tests handle_suggest with no valid targets."""
  with patch("importlib.import_module") as mock_import:
    mock_mod = type("MockMod", (), {})
    mock_import.return_value = mock_mod
    assert handle_suggest("empty_module.*") == 1


def test_handle_suggest_wildcard_exception(capsys):
  """Tests wildcard handling when _extract_metadata raises Exception."""
  with patch("importlib.import_module") as mock_import:
    mock_mod = type("MockMod", (), {})
    mock_mod.func1 = lambda: None
    mock_mod.func2 = lambda: None
    mock_import.return_value = mock_mod

    original_extract = suggest_module._extract_metadata

    def mock_extract(obj):
      if obj is mock_mod.func1:
        raise Exception("Failed")
      return original_extract(obj)

    with patch.object(suggest_module, "_extract_metadata", side_effect=mock_extract):
      assert handle_suggest("my_module.*") == 0
      captured = capsys.readouterr()
      assert "my_module.func2" in captured.out
      assert "my_module.func1" not in captured.out


def test_inspect_live_object():
  """Verifies the behavior of inspect live object."""
  meta = _inspect_live_object("os.path.join")
  assert "signature" in meta


def test_inspect_live_object_invalid_path():
  """Verifies the behavior of inspect live object invalid path."""
  with pytest.raises(ImportError):
    _inspect_live_object("os")


def test_extract_metadata_value_error():
  """Test extract_metadata fallback."""
  import _ast

  res = _extract_metadata(_ast.AST)
  assert res["signature"] == "Unknown Signature"


def test_build_header():
  """Test build header coverage."""
  header = _build_header('{"type": "object"}')
  assert "OUTPUT FORMAT" in header


def test_build_footer():
  """Test build footer coverage."""
  footer = _build_footer("torch")
  assert "INSTRUCTIONS" in footer
  assert "torch" in footer


def test_build_target_block():
  """Test build target block coverage."""
  info = {"kind": "function", "signature": "()", "docstring": "Line 1\nLine 2"}
  block = _build_target_block("a.b.c", info)
  assert "Name: a.b.c" in block
  assert "Line 1\n>Line 2" in block


def test_handle_suggest_outdir_exists(tmp_path):
  """Tests handle_suggest with out_dir that already exists."""
  out_dir = tmp_path / "out"
  out_dir.mkdir(parents=True, exist_ok=True)
  with patch("importlib.import_module") as mock_import:
    mock_mod = type("MockMod", (), {})
    mock_mod.func1 = lambda: None
    mock_import.return_value = mock_mod
    assert handle_suggest("my_module.*", out_dir=out_dir, batch_size=1) == 0
    files = list(out_dir.glob("*.md"))
    assert len(files) == 1


def test_handle_suggest_wildcard_with_module():
  """Tests handle_suggest skipping modules."""
  import types

  with patch("importlib.import_module") as mock_import:
    mock_mod = types.ModuleType("MockMod")
    mock_mod.submod = types.ModuleType("submod")
    mock_mod.func1 = lambda: None
    mock_import.return_value = mock_mod
    assert handle_suggest("my_module.*") == 0
