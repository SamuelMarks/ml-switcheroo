"""Test suite for the Dummy Plugins module."""

import importlib
from unittest.mock import patch

import ml_switcheroo.plugins


def test_dummy_plugins_init() -> None:
  """Verifies the behavior of dummy plugins initialization."""
  with patch("importlib.import_module", side_effect=Exception("mocked err")):
    importlib.reload(ml_switcheroo.plugins)
