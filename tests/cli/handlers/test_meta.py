"""Test suite for meta."""

from ml_switcheroo.cli.handlers.meta import handle_schema


def test_handle_schema() -> None:
  """Verify handle schema behavior."""
  assert handle_schema() == 0
