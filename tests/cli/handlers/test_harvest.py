"""Unit tests for `handle_harvest` (`ml_switcheroo.cli.handlers.harvest`)."""

from argparse import Namespace
from pathlib import Path

import pytest

from ml_switcheroo.cli.handlers.harvest import handle_harvest


def test_handle_harvest_prints_status(capsys: pytest.CaptureFixture[str]) -> None:
  """Verify `handle_harvest` reports target manual-test path and completion status.

  Args:
      capsys: Pytest stdout/stderr capture fixture.
  """
  handle_harvest(Namespace(path=Path("tests/manual")))
  captured = capsys.readouterr()
  assert "Harvesting mappings from manual tests at:" in captured.out
  assert "Harvest complete." in captured.out
