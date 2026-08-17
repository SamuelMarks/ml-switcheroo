"""Test suite for the Paths Coverage module."""

from unittest.mock import patch, MagicMock
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def test_resolve_semantics_dir_local(tmp_path):
  """Resolves semantics a directory local."""
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = True
    assert resolve_semantics_dir() == mock_instance


def test_resolve_semantics_dir_installed(tmp_path):
  """Resolves semantics a directory installed."""
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = False
    with patch("ml_switcheroo.semantics.paths.files", return_value="installed_path"):
      with patch("ml_switcheroo.semantics.paths.sys") as mock_sys:
        mock_sys.version_info = (3, 9)
        resolve_semantics_dir()


def test_resolve_snapshots_dir():
  """Resolves snapshots directory."""
  with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir") as mock_resolve:
    mock_instance = MagicMock()
    mock_resolve.return_value = mock_instance
    resolve_snapshots_dir()
    mock_instance.parent.parent.parent.parent.__truediv__.assert_called_once_with("ml-compiler-snapshots")


def test_resolve_semantics_dir_fallback():
  """Resolves semantics a directory fallback."""
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = False
    with patch("ml_switcheroo.semantics.paths.files", side_effect=Exception("Failed")):
      res = resolve_semantics_dir()
      assert res == mock_instance
