"""Test suite for CLI extra coverage."""

from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from ml_switcheroo.cli.commands import handle_gen_weight_script
from ml_switcheroo.cli.handlers.dev import handle_matrix, handle_docs, handle_gen_tests


def test_gen_weight_script_failure():
  """Test generation of weight script failure."""
  # Coverage for failure in commands.py handle_gen_weight_script
  with patch("ml_switcheroo.cli.commands.WeightScriptGenerator.generate", return_value=False):
    assert handle_gen_weight_script(Path("in.py"), Path("out.py")) == 1


def test_dev_handlers():
  """Test developer handlers."""
  # Cover dev handler operations
  with patch("ml_switcheroo.cli.handlers.dev.CompatibilityMatrix") as MockMatrix:
    mock_matrix = MagicMock()
    MockMatrix.return_value = mock_matrix
    assert handle_matrix() == 0
    mock_matrix.render.assert_called_once()

  with (
    patch("ml_switcheroo.cli.handlers.dev.MigrationGuideGenerator") as MockGen,
    patch("builtins.open", mock_open()),
    patch("ml_switcheroo.cli.handlers.dev.SemanticsManager"),
  ):
    mock_gen_instance = MagicMock()
    mock_gen_instance.generate.return_value = "Docs"
    MockGen.return_value = mock_gen_instance
    assert handle_docs("torch", "jax", Path("docs.md")) == 0
    mock_gen_instance.generate.assert_called_once()

  with (
    patch("ml_switcheroo.cli.handlers.dev.TestCaseGenerator") as MockGen,
    patch("ml_switcheroo.cli.handlers.dev.Path.mkdir"),
  ):
    mock_gen_instance = MagicMock()
    MockGen.return_value = mock_gen_instance
    assert handle_gen_tests(Path("out.py")) == 0
    mock_gen_instance.generate.assert_called_once()
