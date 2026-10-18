"""Test suite for dev."""

from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.cli.handlers.dev import handle_docs, handle_gen_tests, handle_matrix


def test_handle_matrix() -> None:
  """Verify handle matrix behavior."""
  with patch("ml_switcheroo.cli.handlers.dev.CompatibilityMatrix"):
    assert handle_matrix() == 0


def test_handle_docs(tmp_path: Path) -> None:
  """Verify handle docs behavior.

  Args:
      tmp_path: Test argument or fixture.
  """
  with patch("ml_switcheroo.cli.handlers.dev.MigrationGuideGenerator") as MockGen:
    MockGen.return_value.generate.return_value = "markdown"
    out: Path = tmp_path / "docs.md"
    assert handle_docs("torch", "jax", out) == 0
    assert out.exists()


def test_handle_gen_tests(tmp_path: Path) -> None:
  """Verify handle gen tests behavior.

  Args:
      tmp_path: Test argument or fixture.
  """
  with patch("ml_switcheroo.cli.handlers.dev.TestCaseGenerator"):
    out: Path = tmp_path / "tests" / "test.py"
    assert handle_gen_tests(out) == 0
