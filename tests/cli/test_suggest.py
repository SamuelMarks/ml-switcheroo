"""Test module."""

import pytest
import sys
import types
from unittest.mock import patch

from ml_switcheroo.cli.handlers.suggest import (
  handle_suggest,
  _extract_metadata,
  _inspect_live_object,
  _build_header,
  _build_target_block,
  _build_footer,
)


# Dummy objects for testing inspection
def dummy_func(a: int, b: int = 2):
  """Dummy doc."""
  pass


class DummyClass:
  """Class doc."""

  def __init__(self, x):
    """Test element."""
    pass


# Add a dummy module to sys.modules

dummy_mod = types.ModuleType("dummy_mod")
dummy_mod.dummy_func = dummy_func
dummy_mod.DummyClass = DummyClass
sys.modules["dummy_mod"] = dummy_mod


def test_extract_metadata():
  """Test element."""
  info = _extract_metadata(dummy_func)
  assert info["kind"] == "function"
  assert "Dummy doc." in info["docstring"]
  assert "a: int" in info["signature"]

  info_cls = _extract_metadata(DummyClass)
  assert info_cls["kind"] == "class"
  assert "Class doc." in info_cls["docstring"]


def test_extract_metadata_no_sig():
  """Test element."""
  # Builtins often don't have standard signatures inspectable via inspect.signature
  info = _extract_metadata(print)
  assert info["kind"] == "function"
  assert "Unknown Signature" in info["signature"]  # Because ValueError/TypeError is caught


def test_inspect_live_object():
  """Test element."""
  info = _inspect_live_object("dummy_mod.dummy_func")
  assert info["kind"] == "function"

  with pytest.raises(ImportError):
    _inspect_live_object("no_dot_here")

  with pytest.raises(AttributeError):
    _inspect_live_object("dummy_mod.missing")


def test_build_blocks():
  """Test element."""
  header = _build_header("{}")
  assert "You are an expert AI assistant" in header

  info = {"kind": "function", "signature": "(x)", "docstring": "docs"}
  block = _build_target_block("pkg.mod.func", info)
  assert "Name: pkg.mod.func" in block
  assert "func(x)" in block

  footer = _build_footer("torch")
  assert "source framework ('torch')" in footer


def test_handle_suggest_single(capsys):
  """Test element."""
  res = handle_suggest("dummy_mod.dummy_func")
  assert res == 0
  captured = capsys.readouterr()
  assert "You are an expert AI assistant" in captured.out
  assert "Name: dummy_mod.dummy_func" in captured.out
  assert "--- INSTRUCTIONS ---" in captured.out


def test_handle_suggest_single_fail():
  """Test element."""
  res = handle_suggest("dummy_mod.missing_func")
  assert res == 1


def test_handle_suggest_wildcard(tmp_path):
  """Test element."""
  out_dir = tmp_path / "out"
  res = handle_suggest("dummy_mod.*", out_dir=out_dir, batch_size=1)
  assert res == 0

  # Check that files were created
  assert out_dir.exists()
  files = list(out_dir.glob("*.md"))
  assert len(files) == 2  # One for func, one for class

  content = files[0].read_text()
  assert "You are an expert" in content


def test_handle_suggest_wildcard_fail():
  """Test element."""
  res = handle_suggest("missing_module.*")
  assert res == 1


def test_handle_suggest_wildcard_empty():
  """Test element."""
  empty_mod = types.ModuleType("empty_mod")
  empty_mod._hidden = 1
  sys.modules["empty_mod"] = empty_mod
  res = handle_suggest("empty_mod.*")
  assert res == 1  # No valid targets


def test_handle_suggest_wildcard_skip_module():
  """Test element."""
  import os

  mod = types.ModuleType("skip_mod")
  mod.submod = os
  sys.modules["skip_mod"] = mod

  res = handle_suggest("skip_mod.*")
  assert res == 1  # only submod exists, it gets skipped, targets is empty


def test_handle_suggest_wildcard_extract_exception():
  """Test element."""
  mod = types.ModuleType("fail_mod")

  # Create an object that raises an exception when passed to inspect.getdoc
  class BadObj:
    pass

  bad = BadObj()

  with patch("ml_switcheroo.cli.handlers.suggest._extract_metadata", side_effect=Exception("Boom")):
    mod.bad = bad
    sys.modules["fail_mod"] = mod

    res = handle_suggest("fail_mod.*")
    assert res == 1  # Extract fails, targets empty
