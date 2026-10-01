"""Unit tests for scripts/test_wasm_ghost.py."""

from pathlib import Path
import runpy
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

from scripts.test_wasm_ghost import (
  TestWasmGhostMode,
  check_forbidden_imports,
  main,
)


def test_check_forbidden_imports_clean() -> None:
  """Test check_forbidden_imports when none of the forbidden libraries are imported."""
  # Custom clean list of libraries not in sys.modules
  clean_libs = ["non_existent_ml_lib_1", "non_existent_ml_lib_2"]
  assert check_forbidden_imports(clean_libs) is True


def test_check_forbidden_imports_detected() -> None:
  """Test check_forbidden_imports detects an imported library with a __file__ attribute."""
  fake_module = ModuleType("fake_forbidden")
  fake_module.__file__ = "/path/to/fake_forbidden/__init__.py"

  with patch.dict(sys.modules, {"fake_forbidden": fake_module}):
    assert check_forbidden_imports(["fake_forbidden"]) is False


def test_check_forbidden_imports_default_list() -> None:
  """Test check_forbidden_imports with default parameter list."""
  with patch("sys.modules", {}):
    assert check_forbidden_imports() is True


def test_test_wasm_ghost_mode_case() -> None:
  """Test TestWasmGhostMode execution directly."""
  case = TestWasmGhostMode()
  case.test_ghost_mode_loads_snapshots()


def test_test_wasm_ghost_main_success() -> None:
  """Test main execution function succeeding."""
  with patch("scripts.test_wasm_ghost.check_forbidden_imports", return_value=True):
    with patch("unittest.TextTestRunner.run") as mock_run:
      mock_res = MagicMock()
      mock_res.wasSuccessful.return_value = True
      mock_run.return_value = mock_res
      assert main() == 0


def test_test_wasm_ghost_main_forbidden_fails() -> None:
  """Test main execution failing when forbidden library is present."""
  with patch("scripts.test_wasm_ghost.check_forbidden_imports", return_value=False):
    assert main() == 1


def test_test_wasm_ghost_main_test_failure() -> None:
  """Test main execution when tests fail."""
  with patch("scripts.test_wasm_ghost.check_forbidden_imports", return_value=True):
    with patch("unittest.TextTestRunner.run") as mock_run:
      mock_res = MagicMock()
      mock_res.wasSuccessful.return_value = False
      mock_run.return_value = mock_res
      assert main() == 1


def test_test_wasm_ghost_entrypoint(monkeypatch: Any) -> None:
  """Test executing scripts/test_wasm_ghost.py as __main__ module.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  import pytest

  repo_root = Path(__file__).resolve().parent.parent
  src_dir = str(repo_root / "src")

  # Remove src_dir from sys.path to hit the `if str(src_path) not in sys.path:` branch
  monkeypatch.setattr(sys, "path", [p for p in sys.path if p != src_dir])
  monkeypatch.setattr("sys.argv", ["scripts/test_wasm_ghost.py"])

  with pytest.raises(SystemExit) as exc_info:
    runpy.run_path(str(repo_root / "scripts" / "test_wasm_ghost.py"), run_name="__main__")
  assert exc_info.value.code == 0
