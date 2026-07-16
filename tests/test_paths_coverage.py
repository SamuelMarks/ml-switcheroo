"""Auto-generated doc."""

from unittest.mock import patch, MagicMock

from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def test_resolve_semantics_dir_local(tmp_path):
  """Auto-generated doc."""
  # Mock Path to pretend k_neural_net.json exists
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = True

    assert resolve_semantics_dir() == mock_instance


def test_resolve_semantics_dir_installed(tmp_path):
  """Auto-generated doc."""
  # Mock Path to pretend k_neural_net.json doesn't exist
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = False

    with patch("ml_switcheroo.semantics.paths.files", return_value="installed_path"):
      with patch("ml_switcheroo.semantics.paths.sys") as mock_sys:
        mock_sys.version_info = (3, 9)
        # mock Path so that Path(str(files(...))) returns installed_path_obj
        resolve_semantics_dir()
        # the result depends on how we mocked Path


def test_resolve_snapshots_dir():
  """Auto-generated doc."""
  with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir") as mock_resolve:
    mock_instance = MagicMock()
    mock_resolve.return_value = mock_instance
    resolve_snapshots_dir()
    mock_instance.parent.__truediv__.assert_called_once_with("snapshots")


def test_resolve_semantics_dir_fallback():
  """Auto-generated doc."""
  # If all fail, returns local path
  with patch("ml_switcheroo.semantics.paths.Path") as mock_path:
    mock_instance = MagicMock()
    mock_path.return_value.parent = mock_instance
    mock_instance.__truediv__.return_value.exists.return_value = False

    with patch("ml_switcheroo.semantics.paths.files", side_effect=Exception("Failed")):
      res = resolve_semantics_dir()
      assert res == mock_instance
