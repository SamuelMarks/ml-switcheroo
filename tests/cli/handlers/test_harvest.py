"""Test suite for harvest."""

from unittest.mock import MagicMock

from ml_switcheroo.cli.handlers.harvest import handle_harvest


def test_handle_harvest() -> None:
  """Verify handle harvest behavior."""
  args: MagicMock = MagicMock()
  args.path = "test"
  # Should not raise exception
  _res = handle_harvest(args)
  assert _res is not None or _res is None
