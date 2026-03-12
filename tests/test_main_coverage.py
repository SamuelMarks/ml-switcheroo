import sys
import pytest
from unittest.mock import patch


def test_main_execution():
  with patch("ml_switcheroo.cli.__main__.main", return_value=0) as mock_main:
    with patch.object(sys, "exit") as mock_exit:
      with patch("ml_switcheroo.__main__.__name__", "__main__"):
        # We need to re-execute the module code as if it was '__main__'
        import runpy

        runpy.run_module("ml_switcheroo.__main__", run_name="__main__")
        mock_exit.assert_called_once_with(0)
