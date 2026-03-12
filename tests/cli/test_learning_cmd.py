import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_switcheroo.cli.handlers.learning import handle_wizard, handle_harvest


def test_handle_wizard():
  with (
    patch("ml_switcheroo.cli.handlers.learning.SemanticsManager"),
    patch("ml_switcheroo.cli.handlers.learning.MappingWizard") as MockWizard,
  ):
    res = handle_wizard("torch")
    assert res == 0
    MockWizard.return_value.start.assert_called_once_with("torch")


def test_handle_harvest_file():
  with (
    patch("ml_switcheroo.cli.handlers.learning.SemanticsManager"),
    patch("ml_switcheroo.cli.handlers.learning.SemanticHarvester") as MockHarvester,
    patch("pathlib.Path.is_file", return_value=True),
    patch("pathlib.Path.is_dir", return_value=False),
  ):
    MockHarvester.return_value.harvest_file.return_value = 1
    res = handle_harvest(Path("test.py"), "jax", False)
    assert res == 0


def test_handle_harvest_dir():
  with (
    patch("ml_switcheroo.cli.handlers.learning.SemanticsManager"),
    patch("ml_switcheroo.cli.handlers.learning.SemanticHarvester") as MockHarvester,
    patch("pathlib.Path.is_file", return_value=False),
    patch("pathlib.Path.is_dir", return_value=True),
    patch("pathlib.Path.rglob", return_value=[Path("test_1.py")]),
  ):
    MockHarvester.return_value.harvest_file.return_value = 0
    res = handle_harvest(Path("tests"), "jax", False)
    assert res == 0


def test_handle_harvest_invalid():
  with (
    patch("ml_switcheroo.cli.handlers.learning.SemanticsManager"),
    patch("ml_switcheroo.cli.handlers.learning.SemanticHarvester"),
    patch("pathlib.Path.is_file", return_value=False),
    patch("pathlib.Path.is_dir", return_value=False),
  ):
    res = handle_harvest(Path("bad"), "jax", False)
    assert res == 1
