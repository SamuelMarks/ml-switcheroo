"""Test suite for the Main module."""

import pytest
from unittest.mock import patch, mock_open
from ml_switcheroo.cli.__main__ import main


def mock_run_verified_pipeline(code):
  """Provides a mock run verified pipeline for testing."""
  return {"status": "success"}


def test_main_schema():
  """Verifies the behavior of main schema."""
  with patch("ml_switcheroo.cli.__main__.handle_schema") as mock_schema:
    mock_schema.return_value = 0
    assert main(["schema"]) == 0
    mock_schema.assert_called_once()


def test_main_scaffold():
  """Verifies the behavior of main scaffold."""
  with patch("ml_switcheroo.cli.__main__.handle_scaffold") as mock_scaffold:
    assert main(["scaffold", "jax.numpy"]) == 0
    mock_scaffold.assert_called_once()


def test_main_harvest():
  """Verifies the behavior of main harvest."""
  with patch("ml_switcheroo.cli.__main__.handle_harvest") as mock_harvest:
    assert main(["harvest", "some_path"]) == 0
    mock_harvest.assert_called_once()


def test_main_verified_pipeline_success():
  """Verifies the behavior of main verified pipeline successfully."""
  with (
    patch("builtins.open", mock_open(read_data="x = 1")),
    patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline") as mock_pipeline,
    patch("sys.stdout"),
  ):
    mock_pipeline.return_value = {"status": "success"}
    assert main(["verified-pipeline", "some_path.py"]) == 0


def test_main_verified_pipeline_failure():
  """Verifies the behavior of main verified pipeline successfully handling failure."""
  with (
    patch("builtins.open", mock_open(read_data="x = 1")),
    patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline") as mock_pipeline,
    patch("sys.stdout"),
  ):
    mock_pipeline.return_value = {"status": "error"}
    assert main(["verified-pipeline", "some_path.py"]) == 1


def test_main_fallback():
  """Verifies the behavior of main fallback."""
  with patch("sys.argv", ["ml_switcheroo"]):
    with pytest.raises(SystemExit):
      main([])
