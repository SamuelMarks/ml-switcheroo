"""Test suite for the Main Entry module."""

import pytest
from unittest.mock import patch


def test_main_entry():
  """Verifies the behavior of main entry."""
  import runpy
  import sys

  with patch.object(sys, "argv", ["ml_switcheroo", "--help"]):
    with pytest.raises(SystemExit) as e:
      runpy.run_module("ml_switcheroo.__main__", run_name="__main__")
    assert e.value.code == 0

  with patch.object(sys, "argv", ["ml_switcheroo", "--help"]):
    with pytest.raises(SystemExit) as e:
      runpy.run_module("ml_switcheroo.cli.__main__", run_name="__main__")
    assert e.value.code == 0


def test_main_entry_not_main():
  """Verifies the behavior when not run as __main__."""
  import runpy

  with patch("ml_switcheroo.cli.__main__.main") as mock_main:
    # Running with run_name != "__main__"
    runpy.run_module("ml_switcheroo.__main__", run_name="not_main")
    mock_main.assert_not_called()

  with patch("ml_switcheroo.cli.__main__.main") as mock_main:
    runpy.run_module("ml_switcheroo.cli.__main__", run_name="not_main")
    mock_main.assert_not_called()
