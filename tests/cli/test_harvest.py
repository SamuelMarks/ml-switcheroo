"""Test module."""

import pytest
from argparse import Namespace
from pathlib import Path
from ml_switcheroo.cli.handlers.harvest import handle_harvest


def test_handle_harvest(capsys: pytest.CaptureFixture[str]) -> None:
  """Test element."""
  args = Namespace(path=Path("tests/manual"))
  handle_harvest(args)
  captured = capsys.readouterr()
  assert "Harvesting mappings from manual tests at:" in captured.out
  assert "Harvest complete." in captured.out
