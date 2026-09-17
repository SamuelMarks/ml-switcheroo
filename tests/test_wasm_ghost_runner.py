"""Tests for scripts/test_wasm_ghost.py."""

from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch
import pytest

import scripts.test_wasm_ghost as wasm_ghost


def test_check_forbidden_imports(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test checking forbidden library imports.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  # When no forbidden packages are present or they have no __file__
  fake_modules = dict(sys.modules)
  for pkg in wasm_ghost.FORBIDDEN_LIBS:
    fake_modules.pop(pkg, None)
  monkeypatch.setattr(sys, "modules", fake_modules)

  assert wasm_ghost.check_forbidden_imports() is True

  # When a forbidden package is present with a __file__
  dummy_mod = ModuleType("torch")
  dummy_mod.__file__ = "/path/to/torch/__init__.py"
  fake_modules["torch"] = dummy_mod

  assert wasm_ghost.check_forbidden_imports() is False

  # When passing custom sequence
  assert wasm_ghost.check_forbidden_imports(["custom_pkg"]) is True


def test_wasm_ghost_mode_test_case() -> None:
  """Test TestWasmGhostMode test execution directly."""
  case = wasm_ghost.TestWasmGhostMode()
  case.test_ghost_mode_loads_snapshots()


def test_main_cli_branches(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test main CLI entrypoint branches in test_wasm_ghost.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  # Branch 1: check_forbidden_imports fails
  with patch.object(wasm_ghost, "check_forbidden_imports", return_value=False):
    exit_code = wasm_ghost.main([])
    assert exit_code == 1

  # Branch 2: check_forbidden_imports passes, suite passes
  mock_result = MagicMock()
  mock_result.wasSuccessful.return_value = True
  with (
    patch.object(wasm_ghost, "check_forbidden_imports", return_value=True),
    patch("scripts.test_wasm_ghost.unittest.TextTestRunner.run", return_value=mock_result),
  ):
    exit_code = wasm_ghost.main([])
    assert exit_code == 0

  # Branch 3: check_forbidden_imports passes, suite fails
  mock_result.wasSuccessful.return_value = False
  with (
    patch.object(wasm_ghost, "check_forbidden_imports", return_value=True),
    patch("scripts.test_wasm_ghost.unittest.TextTestRunner.run", return_value=mock_result),
  ):
    exit_code = wasm_ghost.main([])
    assert exit_code == 1


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test __main__ execution block.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  import importlib
  import runpy

  orig_sys_path = list(sys.path)
  src_str = str(wasm_ghost.src_path)
  sys.path = [p for p in sys.path if p != src_str]

  fake_modules = dict(sys.modules)
  for pkg in wasm_ghost.FORBIDDEN_LIBS:
    fake_modules.pop(pkg, None)
  monkeypatch.setattr(sys, "modules", fake_modules)

  mock_res = MagicMock()
  mock_res.wasSuccessful.return_value = True

  try:
    importlib.reload(wasm_ghost)
    with patch("scripts.test_wasm_ghost.unittest.TextTestRunner.run", return_value=mock_res):
      with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(Path(wasm_ghost.__file__).resolve()), run_name="__main__")
      assert excinfo.value.code == 0
  finally:
    sys.path = orig_sys_path
