"""Docstring."""

from unittest.mock import patch
from ml_switcheroo.cli.handlers.dev import handle_matrix, handle_docs, handle_gen_tests


def test_handle_matrix():
  """Docstring."""
  with patch("ml_switcheroo.cli.handlers.dev.CompatibilityMatrix"):
    assert handle_matrix() == 0


def test_handle_docs(tmp_path):
  """Docstring."""
  with patch("ml_switcheroo.cli.handlers.dev.MigrationGuideGenerator") as MockGen:
    MockGen.return_value.generate.return_value = "markdown"
    out = tmp_path / "docs.md"
    assert handle_docs("torch", "jax", out) == 0
    assert out.exists()


def test_handle_gen_tests(tmp_path):
  """Docstring."""
  with patch("ml_switcheroo.cli.handlers.dev.TestCaseGenerator"):
    out = tmp_path / "tests" / "test.py"
    assert handle_gen_tests(out) == 0
