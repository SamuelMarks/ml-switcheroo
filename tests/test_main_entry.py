"""Test suite for the Main Entry module."""

import pytest
from unittest.mock import patch


def test_main_entry():
  """Verifies the behavior of main entry."""
  import runpy

  with patch("ml_switcheroo.cli.__main__.main", return_value=0) as mock_main:
    with pytest.raises(SystemExit) as e:
      runpy.run_module("ml_switcheroo.__main__", run_name="__main__")
    assert e.value.code == 0
    mock_main.assert_called_once()
