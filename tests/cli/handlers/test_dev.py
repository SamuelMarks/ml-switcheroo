"""Unit tests for developer CLI handlers (`ml_switcheroo.cli.handlers.dev`)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo.cli.handlers.dev import handle_docs, handle_gen_tests, handle_matrix


def test_handle_matrix_docs_and_gen_tests(tmp_path: Path) -> None:
  """Verify `handle_matrix`, `handle_docs`, and `handle_gen_tests` write expected outputs to `tmp_path`.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  with patch("ml_switcheroo.cli.handlers.dev.CompatibilityMatrix") as mock_matrix_cls:
    assert handle_matrix() == 0
    mock_matrix_cls.return_value.render.assert_called_once()

  with patch("ml_switcheroo.cli.handlers.dev.MigrationGuideGenerator") as mock_guide_cls:
    mock_guide_cls.return_value.generate.return_value = "# Migration Guide"
    docs_out = tmp_path / "MIGRATION.md"
    assert handle_docs("torch", "jax", docs_out) == 0
    mock_guide_cls.return_value.generate.assert_called_once_with("torch", "jax")
    assert docs_out.read_text(encoding="utf-8") == "# Migration Guide"

  with (
    patch("ml_switcheroo.cli.handlers.dev.SemanticsManager") as mock_sm_cls,
    patch("ml_switcheroo.cli.handlers.dev.TestCaseGenerator") as mock_gen_cls,
  ):
    mock_sm: MagicMock = mock_sm_cls.return_value
    mock_sm.get_known_apis.return_value = {"Abs": {}}
    tests_out = tmp_path / "generated" / "test_ops.py"
    assert handle_gen_tests(tests_out) == 0
    assert tests_out.parent.exists()
    mock_gen_cls.return_value.generate.assert_called_once_with({"Abs": {}}, tests_out)
