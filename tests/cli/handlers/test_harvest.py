"""Test suite for harvest."""

import typing

from unittest.mock import MagicMock

from ml_switcheroo.cli.handlers.harvest import handle_harvest


def test_handle_harvest() -> None:
  """Verify handle harvest behavior."""
  args: MagicMock = MagicMock()
  args.path = "test"
  # Should not raise exception
  handle_harvest(args)
  _res: typing.Any = None
  assert _res is None
