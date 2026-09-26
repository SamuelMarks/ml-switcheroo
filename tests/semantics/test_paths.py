"""Test suite for the Paths module."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import ml_switcheroo.semantics.paths as paths
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def test_resolve_semantics_dir_local(tmp_path: Path) -> None:
  """Resolves semantics directory locally when odl or k_neural_net exists.

  Args:
      tmp_path: Temporary directory fixture.
  """
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    (tmp_path / "odl").mkdir()
    assert resolve_semantics_dir() == tmp_path


def test_resolve_semantics_dir_fallback(tmp_path: Path) -> None:
  """Resolves semantics directory falling back to files() package resources.

  Args:
      tmp_path: Temporary directory fixture.
  """
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    if sys.version_info >= (3, 9):
      with patch("ml_switcheroo.semantics.paths.files") as mock_files:
        mock_files.return_value = "mock_path"
        assert resolve_semantics_dir() == Path("mock_path")


def test_resolve_semantics_dir_fallback_exception(tmp_path: Path) -> None:
  """Resolves semantics directory falling back correctly when handling an exception.

  Args:
      tmp_path: Temporary directory fixture.
  """
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    if sys.version_info >= (3, 9):
      with patch("ml_switcheroo.semantics.paths.files") as mock_files:
        mock_files.side_effect = Exception("Test Exception")
        assert resolve_semantics_dir() == tmp_path


def test_resolve_semantics_dir_no_files(tmp_path: Path) -> None:
  """Resolves semantics directory when files API is unavailable.

  Args:
      tmp_path: Temporary directory fixture.
  """
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    with patch("ml_switcheroo.semantics.paths.sys") as mock_sys:
      mock_sys.version_info = (3, 8)
      assert resolve_semantics_dir() == tmp_path


def test_resolve_snapshots_dir_priority_env_ecosystem(tmp_path: Path) -> None:
  """Resolves snapshots directory from ML_ECOSYSTEM_SNAPSHOTS_DIR first.

  Args:
      tmp_path: Temporary directory fixture.
  """
  env_dir = tmp_path / "custom_ecosystem_snapshots"
  env_dir.mkdir()
  with patch.dict(os.environ, {"ML_ECOSYSTEM_SNAPSHOTS_DIR": str(env_dir)}):
    assert resolve_snapshots_dir() == env_dir


def test_resolve_snapshots_dir_env_ecosystem_nonexistent(tmp_path: Path) -> None:
  """Ignores ML_ECOSYSTEM_SNAPSHOTS_DIR if the directory does not exist.

  Args:
      tmp_path: Temporary directory fixture.
  """
  non_existent = tmp_path / "non_existent_eco"
  fake_home = tmp_path / "home"
  with patch.dict(os.environ, {"ML_ECOSYSTEM_SNAPSHOTS_DIR": str(non_existent)}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir") as mock_resolve:
        mock_resolve.return_value = tmp_path / "src" / "ml_switcheroo" / "semantics"
        # Should not resolve to non_existent
        res = resolve_snapshots_dir()
        assert res != non_existent


def test_resolve_snapshots_dir_sibling_ecosystem(tmp_path: Path) -> None:
  """Resolves snapshots from sibling ml-ecosystem-snapshots directory.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root = tmp_path / "repos"
  sem_dir = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  eco_snapshots = repos_root / "ml-ecosystem-snapshots" / "src" / "ml_ecosystem_snapshots" / "snapshots"
  eco_snapshots.mkdir(parents=True)

  fake_home = tmp_path / "empty_home"
  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == eco_snapshots


def test_resolve_snapshots_dir_sibling_framework(tmp_path: Path) -> None:
  """Resolves snapshots from sibling ml-ecosystem-snapshots ml_framework_snapshots path.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root = tmp_path / "repos"
  sem_dir = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fw_snapshots = repos_root / "ml-ecosystem-snapshots" / "src" / "ml_framework_snapshots" / "snapshots"
  fw_snapshots.mkdir(parents=True)

  fake_home = tmp_path / "empty_home"
  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == fw_snapshots


def test_resolve_snapshots_dir_user_cache(tmp_path: Path) -> None:
  """Resolves snapshots from user cache directory ~/.cache/ml_ecosystem_snapshots/snapshots.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root = tmp_path / "repos"
  sem_dir = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fake_home = tmp_path / "home"
  cache_snap = fake_home / ".cache" / "ml_ecosystem_snapshots" / "snapshots"
  cache_snap.mkdir(parents=True)

  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == cache_snap


def test_resolve_snapshots_dir_user_cache_dir(tmp_path: Path) -> None:
  """Resolves snapshots from user cache directory ~/.cache/ml_ecosystem_snapshots.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root = tmp_path / "repos"
  sem_dir = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fake_home = tmp_path / "home"
  cache_dir = fake_home / ".cache" / "ml_ecosystem_snapshots"
  cache_dir.mkdir(parents=True)

  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == cache_dir


def test_resolve_snapshots_dir_env_framework(tmp_path: Path) -> None:
  """Resolves snapshots from legacy ML_FRAMEWORK_SNAPSHOTS_DIR.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root = tmp_path / "repos"
  sem_dir = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  env_dir = tmp_path / "custom_fw_snapshots"
  env_dir.mkdir()
  fake_home = tmp_path / "empty_home"

  with patch.dict(os.environ, {"ML_FRAMEWORK_SNAPSHOTS_DIR": str(env_dir)}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == env_dir


def test_resolve_snapshots_dir_env_framework_nonexistent(tmp_path: Path) -> None:
  """Ignores ML_FRAMEWORK_SNAPSHOTS_DIR if directory does not exist.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root = tmp_path / "repos"
  sem_dir = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  non_existent = tmp_path / "non_existent_fw"
  fake_home = tmp_path / "empty_home"

  with patch.dict(os.environ, {"ML_FRAMEWORK_SNAPSHOTS_DIR": str(non_existent)}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        res = resolve_snapshots_dir()
        assert res != non_existent


def test_resolve_snapshots_dir_legacy_compiler(tmp_path: Path) -> None:
  """Resolves snapshots from legacy ml-compiler-snapshots directory.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root = tmp_path / "repos"
  sem_dir = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  compiler_cand = repos_root / "ml-compiler-snapshots"
  compiler_cand.mkdir(parents=True)
  fake_home = tmp_path / "empty_home"

  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == compiler_cand


def test_resolve_snapshots_dir_legacy_framework(tmp_path: Path) -> None:
  """Resolves snapshots from legacy ml-framework-snapshots directory.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root = tmp_path / "repos"
  sem_dir = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fw_cand = repos_root / "ml-framework-snapshots" / "src" / "ml_framework_snapshots" / "snapshots"
  fw_cand.mkdir(parents=True)
  fake_home = tmp_path / "empty_home"

  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == fw_cand


def test_resolve_snapshots_dir_fallback(tmp_path: Path) -> None:
  """Resolves snapshots fallback when none exists.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root = tmp_path / "repos"
  sem_dir = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fake_home = tmp_path / "empty_home"

  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        expected = repos_root / "ml-compiler-snapshots"
        assert resolve_snapshots_dir() == expected


def test_python_old() -> None:
  """Verifies the behavior of python old."""
  import importlib

  with patch("sys.version_info", (3, 8)):
    importlib.reload(paths)
    assert paths.files is None
  importlib.reload(paths)
