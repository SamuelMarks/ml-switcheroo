"""Test suite for the Main Entry module."""

import pytest
from unittest.mock import patch
import sys


def test_main_entry():
  """Verifies the behavior of main entry."""
  import runpy

  sys.modules.pop("ml_switcheroo.__main__", None)
  with patch.object(sys, "argv", ["ml_switcheroo", "--help"]):
    with pytest.raises(SystemExit) as e:
      runpy.run_module("ml_switcheroo.__main__", run_name="__main__")
    assert e.value.code == 0

  sys.modules.pop("ml_switcheroo.cli.__main__", None)
  with patch.object(sys, "argv", ["ml_switcheroo", "--help"]):
    with pytest.raises(SystemExit) as e:
      runpy.run_module("ml_switcheroo.cli.__main__", run_name="__main__")
    assert e.value.code == 0


def test_main_entry_define(tmp_path):
  """Verifies the behavior of main entry for the define command."""
  import ml_switcheroo.cli.__main__

  with patch("ml_switcheroo.cli.__main__.commands.handle_define") as mock_define:
    mock_define.return_value = 0
    with patch.object(sys, "argv", ["ml_switcheroo", "define", str(tmp_path / "test.yaml")]):
      try:
        assert ml_switcheroo.cli.__main__.main() == 0
      except SystemExit as e:
        assert e.code == 0
    mock_define.assert_called_once()


def test_main_entry_not_main():
  """Verifies the behavior when not run as __main__."""
  import runpy

  sys.modules.pop("ml_switcheroo.__main__", None)
  with patch("ml_switcheroo.cli.__main__.main") as mock_main:
    # Running with run_name != "__main__"
    runpy.run_module("ml_switcheroo.__main__", run_name="not_main")
    mock_main.assert_not_called()

  sys.modules.pop("ml_switcheroo.cli.__main__", None)
  with patch("ml_switcheroo.cli.__main__.main") as mock_main:
    runpy.run_module("ml_switcheroo.cli.__main__", run_name="not_main")
    mock_main.assert_not_called()
