"""Unit tests for `handle_suggest` and live object metadata extraction (`ml_switcheroo.cli.handlers.suggest`)."""

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from ml_switcheroo.cli.handlers.suggest import (
  _build_footer,
  _build_header,
  _build_target_block,
  _extract_metadata,
  _inspect_live_object,
  handle_suggest,
)


def _dummy_func(a: int, b: int = 2) -> None:
  """Add parameters `a` and `b`.

  Args:
      a: First integer operand.
      b: Second integer operand.
  """
  _ = (a, b)


class _DummyClass:
  """Sample class with typed `__init__`."""

  def __init__(self, x: int) -> None:
    """Store `x`.

    Args:
        x: Input integer.
    """
    self.x = x


@pytest.fixture(autouse=True)
def _register_dummy_mod(monkeypatch: pytest.MonkeyPatch) -> None:
  """Register a scoped `suggest_dummy_mod` module in `sys.modules`.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  dummy_mod = types.ModuleType("suggest_dummy_mod")
  monkeypatch.setattr(dummy_mod, "dummy_func", _dummy_func, raising=False)
  monkeypatch.setattr(dummy_mod, "DummyClass", _DummyClass, raising=False)
  monkeypatch.setitem(sys.modules, "suggest_dummy_mod", dummy_mod)


def test_extract_metadata_inspect_and_prompt_builders() -> None:
  """Verify `_extract_metadata`, `_inspect_live_object`, and prompt section builders."""
  fn_info = _extract_metadata(_dummy_func)
  assert fn_info["kind"] == "function"
  assert "Add parameters" in fn_info["docstring"]
  assert "a: int" in fn_info["signature"]

  cls_info = _extract_metadata(_DummyClass)
  assert cls_info["kind"] == "class"
  assert "Sample class" in cls_info["docstring"]

  builtin_info = _extract_metadata(iter)
  assert builtin_info["kind"] == "function"
  assert "Unknown Signature" in builtin_info["signature"]

  mod_info = _extract_metadata(os)
  assert mod_info["signature"] == "Unknown Signature"

  live = _inspect_live_object("suggest_dummy_mod.dummy_func")
  assert live is not None
  assert live["kind"] == "function"

  with pytest.raises(ImportError):
    _inspect_live_object("no_dot_here")

  with pytest.raises(AttributeError):
    _inspect_live_object("suggest_dummy_mod.missing")

  assert "You are an expert AI assistant" in _build_header("{}")
  block = _build_target_block("pkg.mod.func", {"kind": "function", "signature": "(x)", "docstring": "docs"})
  assert "Name: pkg.mod.func" in block
  assert "func(x)" in block
  assert "source framework ('torch')" in _build_footer("torch")


def test_handle_suggest_single_and_wildcard_modes(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """Verify `handle_suggest` single-symbol, wildcard-batch, pre-existing output dir, and failure paths.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
      capsys: Pytest stdout/stderr capture fixture.
  """
  assert handle_suggest("suggest_dummy_mod.dummy_func") == 0
  out_txt = capsys.readouterr().out
  assert "Name: suggest_dummy_mod.dummy_func" in out_txt
  assert "--- INSTRUCTIONS ---" in out_txt

  single_dir = tmp_path / "single_out"
  assert handle_suggest("suggest_dummy_mod.dummy_func", out_dir=single_dir) == 0
  assert len(list(single_dir.glob("*.md"))) == 1

  assert handle_suggest("suggest_dummy_mod.missing_func") == 1
  assert handle_suggest("nodots") == 1

  # Wildcard batch creation on fresh and already-existing directory
  wild_dir = tmp_path / "wild_out"
  assert handle_suggest("suggest_dummy_mod.*", out_dir=wild_dir, batch_size=1) == 0
  assert len(list(wild_dir.glob("*.md"))) == 2
  assert handle_suggest("suggest_dummy_mod.*", out_dir=wild_dir, batch_size=10) == 0

  # Wildcard failure modes: missing module, empty module, submodule-only module, and metadata exception
  assert handle_suggest("nonexistent_suggest_pkg_xyz.*") == 1

  empty_mod = types.ModuleType("suggest_empty_mod")
  monkeypatch.setattr(empty_mod, "_hidden", 1, raising=False)
  monkeypatch.setitem(sys.modules, "suggest_empty_mod", empty_mod)
  assert handle_suggest("suggest_empty_mod.*") == 1

  skip_mod = types.ModuleType("suggest_skip_mod")
  monkeypatch.setattr(skip_mod, "submod", os, raising=False)
  monkeypatch.setitem(sys.modules, "suggest_skip_mod", skip_mod)
  assert handle_suggest("suggest_skip_mod.*") == 1

  fail_mod = types.ModuleType("suggest_fail_mod")
  monkeypatch.setattr(fail_mod, "item", _DummyClass(1), raising=False)
  monkeypatch.setitem(sys.modules, "suggest_fail_mod", fail_mod)
  with patch("ml_switcheroo.cli.handlers.suggest._extract_metadata", side_effect=RuntimeError("boom")):
    assert handle_suggest("suggest_fail_mod.*") == 1
