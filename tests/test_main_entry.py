"""Test suite for the Main Entry module."""

import pytest
import sys
from unittest.mock import patch


def test_main_entry():
  """Verifies the behavior of main entry."""
  with patch("ml_switcheroo.__main__.main") as mock_main:
    mock_main.return_value = 0
    with patch.object(sys, "argv", ["ml_switcheroo"]):
      import ml_switcheroo.__main__

      with patch.dict("sys.modules", {"ml_switcheroo.__main__": ml_switcheroo.__main__}):
        with patch("ml_switcheroo.__main__.__name__", "__main__"):
          with pytest.raises(SystemExit) as e:
            sys.exit(ml_switcheroo.__main__.main())
          assert e.value.code == 0
