"""Test suite for the Paths Coverage module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def test_resolve_semantics_dir_local(tmp_path: Path) -> None:
  """Resolves semantics a directory local."""
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance: MagicMock = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = True
    assert resolve_semantics_dir() == mock_instance


def test_resolve_semantics_dir_installed(tmp_path: Path) -> None:
  """Resolves semantics a directory installed."""
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance: MagicMock = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = False
    with patch("ml_switcheroo.semantics.paths.files", return_value="installed_path"):
      with patch("ml_switcheroo.semantics.paths.sys") as mock_sys:
        mock_sys.version_info = (3, 9)
        resolve_semantics_dir()


def test_resolve_snapshots_dir() -> None:
  """Resolves snapshots directory."""
  with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir") as mock_resolve:
    mock_instance: MagicMock = MagicMock()
    mock_resolve.return_value = mock_instance
    mock_candidate = MagicMock()
    mock_candidate.exists.return_value = False
    mock_fw_candidate = MagicMock()
    mock_fw_candidate.exists.return_value = False

    def truediv_side_effect(arg: str) -> MagicMock:
      """Side effect function simulating path division."""
      if arg == "ml-compiler-snapshots":
        return mock_candidate
      if arg == "ml-framework-snapshots":
        sub = MagicMock()
        sub.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_fw_candidate
        return sub
      return MagicMock()

    mock_instance.parent.parent.parent.parent.__truediv__.side_effect = truediv_side_effect
    res = resolve_snapshots_dir()
    assert res == mock_candidate


def test_resolve_snapshots_dir_candidate_exists() -> None:
  """Resolves snapshots directory when ml-compiler-snapshots exists."""
  with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir") as mock_resolve:
    mock_instance: MagicMock = MagicMock()
    mock_resolve.return_value = mock_instance
    mock_candidate = MagicMock()
    mock_candidate.exists.return_value = True
    mock_instance.parent.parent.parent.parent.__truediv__.return_value = mock_candidate

    res = resolve_snapshots_dir()
    assert res == mock_candidate


def test_resolve_snapshots_dir_framework_candidate_exists() -> None:
  """Resolves snapshots directory when ml-framework-snapshots exists."""
  with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir") as mock_resolve:
    mock_instance: MagicMock = MagicMock()
    mock_resolve.return_value = mock_instance
    mock_candidate = MagicMock()
    mock_candidate.exists.return_value = False
    mock_fw_candidate = MagicMock()
    mock_fw_candidate.exists.return_value = True

    def truediv_side_effect(arg: str) -> MagicMock:
      """Side effect function simulating framework path division."""
      if arg == "ml-compiler-snapshots":
        return mock_candidate
      if arg == "ml-framework-snapshots":
        sub = MagicMock()
        sub.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_fw_candidate
        return sub
      return MagicMock()

    mock_instance.parent.parent.parent.parent.__truediv__.side_effect = truediv_side_effect
    res = resolve_snapshots_dir()
    assert res == mock_fw_candidate


def test_resolve_semantics_dir_fallback() -> None:
  """Resolves semantics a directory fallback."""
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance: MagicMock = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = False
    with patch("ml_switcheroo.semantics.paths.files", side_effect=Exception("Failed")):
      res: Path = resolve_semantics_dir()
      assert res == mock_instance
