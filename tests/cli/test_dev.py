"""Test module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_switcheroo.cli.handlers.dev import handle_matrix, handle_docs, handle_gen_tests


@patch("ml_switcheroo.cli.handlers.dev.CompatibilityMatrix")
def test_handle_matrix(mock_matrix_class: MagicMock) -> None:
  """Test element."""
  mock_instance: MagicMock = mock_matrix_class.return_value
  res: int = handle_matrix()
  assert res == 0
  mock_instance.render.assert_called_once()


@patch("ml_switcheroo.cli.handlers.dev.MigrationGuideGenerator")
def test_handle_docs(mock_guide_class: MagicMock, tmp_path: Path) -> None:
  """Test element."""
  mock_instance: MagicMock = mock_guide_class.return_value
  mock_instance.generate.return_value = "# Markdown Output"

  out_file: Path = tmp_path / "MIGRATION.md"
  res: int = handle_docs("torch", "jax", out_file)

  assert res == 0
  mock_instance.generate.assert_called_once_with("torch", "jax")
  assert out_file.read_text() == "# Markdown Output"


@patch("ml_switcheroo.cli.handlers.dev.TestCaseGenerator")
def test_handle_gen_tests(mock_gen_class: MagicMock, tmp_path: Path) -> None:
  """Test element."""
  mock_instance: MagicMock = mock_gen_class.return_value

  out_file: Path = tmp_path / "tests" / "gen.py"

  mock_sm_class: MagicMock
  with patch("ml_switcheroo.cli.handlers.dev.SemanticsManager") as mock_sm_class:
    mock_sm: MagicMock = mock_sm_class.return_value
    mock_sm.get_known_apis.return_value = {"op": {}}

    res: int = handle_gen_tests(out_file)

    assert res == 0
    assert out_file.parent.exists()
    mock_instance.generate.assert_called_once_with({"op": {}}, out_file)
