import pytest
import sys
from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir
import ml_switcheroo.semantics.paths as paths


def test_resolve_semantics_dir_local(tmp_path):
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    (tmp_path / "k_neural_net.json").touch()
    assert resolve_semantics_dir() == tmp_path


def test_resolve_semantics_dir_fallback(tmp_path):
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    # Do not create k_neural_net.json
    if sys.version_info >= (3, 9):
      with patch("ml_switcheroo.semantics.paths.files") as mock_files:
        mock_files.return_value = "mock_path"
        assert resolve_semantics_dir() == Path("mock_path")


def test_resolve_semantics_dir_fallback_exception(tmp_path):
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    if sys.version_info >= (3, 9):
      with patch("ml_switcheroo.semantics.paths.files") as mock_files:
        mock_files.side_effect = Exception("Test Exception")
        assert resolve_semantics_dir() == tmp_path


def test_resolve_semantics_dir_no_files(tmp_path):
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    with patch("ml_switcheroo.semantics.paths.sys") as mock_sys:
      mock_sys.version_info = (3, 8)
      assert resolve_semantics_dir() == tmp_path


def test_resolve_snapshots_dir():
  with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/foo/bar/semantics")
    assert resolve_snapshots_dir() == Path("/foo/bar/snapshots")


def test_python_old():
  # To hit line 15, we need to reload the module with sys.version_info < (3, 9)
  # But reloading can be tricky. Let's patch sys.version_info and reload paths
  import importlib

  with patch("sys.version_info", (3, 8)):
    importlib.reload(paths)
    assert paths.files is None
  # Restore
  importlib.reload(paths)
