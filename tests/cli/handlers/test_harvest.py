"""Docstring."""

from unittest.mock import MagicMock
from ml_switcheroo.cli.handlers.harvest import handle_harvest


def test_handle_harvest():
  """Docstring."""
  args = MagicMock()
  args.path = "test"
  # Should not raise exception
  handle_harvest(args)
