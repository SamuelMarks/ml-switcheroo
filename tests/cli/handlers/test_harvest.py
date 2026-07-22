"""Test suite for the Harvest module."""

from argparse import Namespace
from ml_switcheroo.cli.handlers.harvest import handle_harvest


def test_handle_harvest(capsys):
  """Handles harvest."""
  args = Namespace(path="some_path")
  handle_harvest(args)
  captured = capsys.readouterr()
  assert "Harvesting mappings from manual tests at: some_path" in captured.out
  assert "Harvest complete. ODL definitions updated." in captured.out
