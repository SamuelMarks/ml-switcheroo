"""Docstring."""

from unittest.mock import patch
from pathlib import Path
from ml_switcheroo.cli.handlers.dev import handle_matrix, handle_docs, handle_gen_tests


def test_handle_matrix() -> None:
  """Docstring."""
  with patch("ml_switcheroo.cli.handlers.dev.CompatibilityMatrix"):
    assert handle_matrix() == 0


def test_handle_docs(tmp_path: Path) -> None:
  """Docstring."""
  with patch("ml_switcheroo.cli.handlers.dev.MigrationGuideGenerator") as MockGen:
    MockGen.return_value.generate.return_value = "markdown"
    out: Path = tmp_path / "docs.md"
    assert handle_docs("torch", "jax", out) == 0
    assert out.exists()


def test_handle_gen_tests(tmp_path: Path) -> None:
  """Docstring."""
  with patch("ml_switcheroo.cli.handlers.dev.TestCaseGenerator"):
    out: Path = tmp_path / "tests" / "test.py"
    assert handle_gen_tests(out) == 0
