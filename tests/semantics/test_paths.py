"""Test suite for the Paths module."""

import typing

import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

import ml_switcheroo.semantics.paths as paths  # noqa: E402
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir  # noqa: E402


def test_resolve_semantics_dir_local(tmp_path: typing.Any) -> None:
  """Resolves semantics directory locally when odl or k_neural_net exists.

  Args:
      tmp_path: Temporary directory fixture.
  """
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    (tmp_path / "odl").mkdir()
    assert resolve_semantics_dir() == tmp_path


def test_resolve_semantics_dir_fallback(tmp_path: typing.Any) -> None:
  """Resolves semantics directory falling back to files() package resources.

  Args:
      tmp_path: Temporary directory fixture.
  """
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    if sys.version_info >= (3, 9):
      with patch("ml_switcheroo.semantics.paths.files") as mock_files:
        mock_files.return_value = "mock_path"
        assert resolve_semantics_dir() == Path("mock_path")


def test_resolve_semantics_dir_fallback_exception(tmp_path: typing.Any) -> None:
  """Resolves semantics directory falling back correctly when handling an exception.

  Args:
      tmp_path: Temporary directory fixture.
  """
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    if sys.version_info >= (3, 9):
      with patch("ml_switcheroo.semantics.paths.files") as mock_files:
        mock_files.side_effect = OSError("Test Exception")
        assert resolve_semantics_dir() == tmp_path


def test_resolve_semantics_dir_no_files(tmp_path: typing.Any) -> None:
  """Resolves semantics directory when files API is unavailable.

  Args:
      tmp_path: Temporary directory fixture.
  """
  with patch("ml_switcheroo.semantics.paths.__file__", str(tmp_path / "paths.py")):
    with patch("ml_switcheroo.semantics.paths.sys") as mock_sys:
      mock_sys.version_info = (3, 8)
      assert resolve_semantics_dir() == tmp_path


def test_resolve_snapshots_dir_priority_env_ecosystem(tmp_path: typing.Any) -> None:
  """Resolves snapshots directory from ML_ECOSYSTEM_SNAPSHOTS_DIR first.

  Args:
      tmp_path: Temporary directory fixture.
  """
  env_dir: typing.Any = tmp_path / "custom_ecosystem_snapshots"
  env_dir.mkdir()
  with patch.dict(os.environ, {"ML_ECOSYSTEM_SNAPSHOTS_DIR": str(env_dir)}):
    assert resolve_snapshots_dir() == env_dir


def test_resolve_snapshots_dir_env_ecosystem_nonexistent(tmp_path: typing.Any) -> None:
  """Ignores ML_ECOSYSTEM_SNAPSHOTS_DIR if the directory does not exist.

  Args:
      tmp_path: Temporary directory fixture.
  """
  non_existent: typing.Any = tmp_path / "non_existent_eco"
  fake_home: typing.Any = tmp_path / "home"
  with patch.dict(os.environ, {"ML_ECOSYSTEM_SNAPSHOTS_DIR": str(non_existent)}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir") as mock_resolve:
        mock_resolve.return_value = tmp_path / "src" / "ml_switcheroo" / "semantics"
        # Should not resolve to non_existent
        res: typing.Any = resolve_snapshots_dir()
        assert res != non_existent


def test_resolve_snapshots_dir_sibling_ecosystem(tmp_path: typing.Any) -> None:
  """Resolves snapshots from sibling ml-ecosystem-snapshots directory.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root: typing.Any = tmp_path / "repos"
  sem_dir: typing.Any = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  eco_snapshots: typing.Any = repos_root / "ml-ecosystem-snapshots" / "src" / "ml_ecosystem_snapshots" / "snapshots"
  eco_snapshots.mkdir(parents=True)

  fake_home: typing.Any = tmp_path / "empty_home"
  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == eco_snapshots


def test_resolve_snapshots_dir_sibling_framework(tmp_path: typing.Any) -> None:
  """Resolves snapshots from sibling ml-ecosystem-snapshots ml_framework_snapshots path.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root: typing.Any = tmp_path / "repos"
  sem_dir: typing.Any = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fw_snapshots: typing.Any = repos_root / "ml-ecosystem-snapshots" / "src" / "ml_framework_snapshots" / "snapshots"
  fw_snapshots.mkdir(parents=True)

  fake_home: typing.Any = tmp_path / "empty_home"
  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == fw_snapshots


def test_resolve_snapshots_dir_user_cache(tmp_path: typing.Any) -> None:
  """Resolves snapshots from user cache directory ~/.cache/ml_ecosystem_snapshots/snapshots.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root: typing.Any = tmp_path / "repos"
  sem_dir: typing.Any = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fake_home: typing.Any = tmp_path / "home"
  cache_snap: typing.Any = fake_home / ".cache" / "ml_ecosystem_snapshots" / "snapshots"
  cache_snap.mkdir(parents=True)

  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == cache_snap


def test_resolve_snapshots_dir_user_cache_dir(tmp_path: typing.Any) -> None:
  """Resolves snapshots from user cache directory ~/.cache/ml_ecosystem_snapshots.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root: typing.Any = tmp_path / "repos"
  sem_dir: typing.Any = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fake_home: typing.Any = tmp_path / "home"
  cache_dir: typing.Any = fake_home / ".cache" / "ml_ecosystem_snapshots"
  cache_dir.mkdir(parents=True)

  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == cache_dir


def test_resolve_snapshots_dir_env_framework(tmp_path: typing.Any) -> None:
  """Resolves snapshots from legacy ML_FRAMEWORK_SNAPSHOTS_DIR.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root: typing.Any = tmp_path / "repos"
  sem_dir: typing.Any = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  env_dir: typing.Any = tmp_path / "custom_fw_snapshots"
  env_dir.mkdir()
  fake_home: typing.Any = tmp_path / "empty_home"

  with patch.dict(os.environ, {"ML_FRAMEWORK_SNAPSHOTS_DIR": str(env_dir)}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == env_dir


def test_resolve_snapshots_dir_env_framework_nonexistent(tmp_path: typing.Any) -> None:
  """Ignores ML_FRAMEWORK_SNAPSHOTS_DIR if directory does not exist.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root: typing.Any = tmp_path / "repos"
  sem_dir: typing.Any = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  non_existent: typing.Any = tmp_path / "non_existent_fw"
  fake_home: typing.Any = tmp_path / "empty_home"

  with patch.dict(os.environ, {"ML_FRAMEWORK_SNAPSHOTS_DIR": str(non_existent)}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        res: typing.Any = resolve_snapshots_dir()
        assert res != non_existent


def test_resolve_snapshots_dir_legacy_framework(tmp_path: Path) -> None:
  """Resolves snapshots from sibling ml-framework-snapshots directory.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root: Path = tmp_path / "repos"
  sem_dir: Path = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fw_cand: Path = repos_root / "ml-framework-snapshots" / "src" / "ml_framework_snapshots" / "snapshots"
  fw_cand.mkdir(parents=True)
  fake_home: Path = tmp_path / "empty_home"

  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == fw_cand


def test_resolve_snapshots_dir_fallback(tmp_path: Path) -> None:
  """Resolves snapshots fallback to ml-framework-snapshots when none exists.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root: Path = tmp_path / "repos"
  sem_dir: Path = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fake_home: Path = tmp_path / "empty_home"

  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        expected: Path = repos_root / "ml-framework-snapshots"
        assert resolve_snapshots_dir() == expected


def test_resolve_snapshots_dir_custom_paths(tmp_path: typing.Any) -> None:
  """Resolves snapshots directory from custom snapshots path.

  Args:
      tmp_path: Temporary directory fixture.
  """
  custom_dir: typing.Any = tmp_path / "custom_snapshots_env"
  custom_dir.mkdir()
  non_existent: typing.Any = tmp_path / "does_not_exist"
  with patch(
    "ml_switcheroo.semantics.paths.get_custom_snapshots_paths",
    return_value=[str(non_existent), str(custom_dir)],
  ):
    assert resolve_snapshots_dir() == custom_dir


def test_resolve_snapshots_dir_env_snapshots_path(tmp_path: typing.Any) -> None:
  """Resolves snapshots directory from ML_SNAPSHOTS_PATH environment variable.

  Args:
      tmp_path: Temporary directory fixture.
  """
  custom_dir: typing.Any = tmp_path / "env_snapshots_path"
  custom_dir.mkdir()
  with patch.dict(os.environ, {"ML_SNAPSHOTS_PATH": str(custom_dir)}, clear=True):
    assert resolve_snapshots_dir() == custom_dir


def test_resolve_snapshots_dir_env_snapshots_path_nonexistent(tmp_path: typing.Any) -> None:
  """Continues search when ML_SNAPSHOTS_PATH points to non-existent directory.

  Args:
      tmp_path: Temporary directory fixture.
  """
  non_existent: typing.Any = tmp_path / "non_existent_snapshots_path"
  fallback_dir: typing.Any = tmp_path / "fallback_snapshots"
  fallback_dir.mkdir()
  with patch.dict(
    os.environ, {"ML_SNAPSHOTS_PATH": str(non_existent), "ML_FRAMEWORK_SNAPSHOTS_PATH": str(fallback_dir)}, clear=True
  ):
    assert resolve_snapshots_dir() == fallback_dir


def test_resolve_snapshots_dir_env_framework_snapshots_path(tmp_path: typing.Any) -> None:
  """Resolves snapshots directory from ML_FRAMEWORK_SNAPSHOTS_PATH environment variable.

  Args:
      tmp_path: Temporary directory fixture.
  """
  custom_dir: typing.Any = tmp_path / "env_fw_snapshots_path"
  custom_dir.mkdir()
  with patch.dict(os.environ, {"ML_FRAMEWORK_SNAPSHOTS_PATH": str(custom_dir)}, clear=True):
    assert resolve_snapshots_dir() == custom_dir


def test_resolve_snapshots_dir_sibling_framework_direct(tmp_path: typing.Any) -> None:
  """Resolves snapshots from direct ml-framework-snapshots/snapshots directory.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root: typing.Any = tmp_path / "repos"
  sem_dir: typing.Any = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fw_cand: typing.Any = repos_root / "ml-framework-snapshots" / "snapshots"
  fw_cand.mkdir(parents=True)
  fake_home: typing.Any = tmp_path / "empty_home"

  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        assert resolve_snapshots_dir() == fw_cand


def test_paths_import_fallback() -> None:
  """Verifies the behavior of fallback when ml_ecosystem_snapshots.utils is missing."""
  import importlib

  with patch.dict("sys.modules", {"ml_ecosystem_snapshots.utils": None}):
    importlib.reload(paths)
    assert paths.get_custom_snapshots_paths() == []
  importlib.reload(paths)


def test_python_old() -> None:
  """Verifies the behavior of python old."""
  import importlib

  with patch("sys.version_info", (3, 8)):
    importlib.reload(paths)
    assert paths.files is None
  importlib.reload(paths)
