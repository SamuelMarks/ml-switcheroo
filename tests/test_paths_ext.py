"""Test suite for the Paths Coverage module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def test_resolve_semantics_dir_local(tmp_path: Path) -> None:
  """Resolves semantics directory locally when odl directory exists.

  Args:
      tmp_path: Temporary directory fixture.
  """
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance: MagicMock = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = True
    assert resolve_semantics_dir() == mock_instance


def test_resolve_semantics_dir_installed(tmp_path: Path) -> None:
  """Resolves semantics directory from installed package resources.

  Args:
      tmp_path: Temporary directory fixture.
  """
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance: MagicMock = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = False
    with patch("ml_switcheroo.semantics.paths.files", return_value="installed_path"):
      res = resolve_semantics_dir()
  assert res is not None


def test_resolve_snapshots_dir(tmp_path: Path) -> None:
  """Resolves snapshots directory fallback to ml-ecosystem-snapshots when no candidates exist.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root = tmp_path / "repos"
  sem_dir = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  fake_home = tmp_path / "home"
  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        res = resolve_snapshots_dir()
        assert res == repos_root / "ml-ecosystem-snapshots"


def test_resolve_snapshots_dir_ecosystem_candidate_exists(tmp_path: Path) -> None:
  """Resolves snapshots directory when ml-ecosystem-snapshots exists.

  Args:
      tmp_path: Temporary directory fixture.
  """
  repos_root = tmp_path / "repos"
  sem_dir = repos_root / "ml-switcheroo" / "src" / "ml_switcheroo" / "semantics"
  eco_snap = repos_root / "ml-ecosystem-snapshots" / "src" / "ml_ecosystem_snapshots" / "snapshots"
  eco_snap.mkdir(parents=True)
  fake_home = tmp_path / "home"
  with patch.dict(os.environ, {}, clear=True):
    with patch("pathlib.Path.home", return_value=fake_home):
      with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
        res = resolve_snapshots_dir()
        assert res == eco_snap


def test_resolve_semantics_dir_fallback() -> None:
  """Resolves semantics directory fallback when importlib.resources.files raises OSError."""
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance: MagicMock = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = False
    with patch("ml_switcheroo.semantics.paths.files", side_effect=OSError("Failed")):
      res: Path = resolve_semantics_dir()
      assert res == mock_instance
