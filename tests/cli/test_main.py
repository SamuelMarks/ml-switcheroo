"""Test suite for the Main module."""

import pytest
from unittest.mock import patch, mock_open
from ml_switcheroo.cli.__main__ import main


def mock_run_verified_pipeline(code):
  """Provides a mock run verified pipeline for testing.

  Args:
      code: ...
  """
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


def test_main_convert():
  """Verifies the behavior of main convert."""
  with patch("ml_switcheroo.cli.__main__.commands.handle_convert") as mock_handle:
    mock_handle.return_value = 0
    assert main(["convert", "model.py", "--source", "torch", "--target", "jax"]) == 0
    mock_handle.assert_called_once()


def test_main_gen_weight_script():
  """Verifies the behavior of main gen-weight-script."""
  with patch("ml_switcheroo.cli.__main__.commands.handle_gen_weight_script") as mock_handle:
    mock_handle.return_value = 0
    assert main(["gen-weight-script", "model.py", "--out", "out.py", "--source", "torch", "--target", "jax"]) == 0
    mock_handle.assert_called_once()


def test_main_matrix():
  """Verifies the behavior of main matrix."""
  with patch("ml_switcheroo.cli.__main__.commands.handle_matrix") as mock_handle:
    mock_handle.return_value = 0
    assert main(["matrix"]) == 0
    mock_handle.assert_called_once()


def test_main_suggest():
  """Verifies the behavior of main suggest."""
  with patch("ml_switcheroo.cli.__main__.handle_suggest") as mock_handle:
    mock_handle.return_value = 0
    assert main(["suggest", "torch.nn.Linear"]) == 0
    mock_handle.assert_called_once()


def test_main_ci():
  """Verifies the behavior of main ci."""
  with patch("ml_switcheroo.cli.__main__.commands.handle_ci") as mock_handle:
    mock_handle.return_value = 0
    assert main(["ci"]) == 0
    mock_handle.assert_called_once()


def test_main_fallback():
  """Verifies the behavior of main fallback."""
  with patch("sys.argv", ["ml_switcheroo"]):
    with pytest.raises(SystemExit):
      main([])


def test_main_gen_docs():
  """Verifies the behavior of main gen-docs."""
  with patch("ml_switcheroo.cli.__main__.commands.handle_docs") as mock_handle:
    mock_handle.return_value = 0
    assert main(["gen-docs"]) == 0
    mock_handle.assert_called_once()


def test_main_gen_tests():
  """Verifies the behavior of main gen-tests."""
  with patch("ml_switcheroo.cli.__main__.commands.handle_gen_tests") as mock_handle:
    mock_handle.return_value = 0
    assert main(["gen-tests"]) == 0
    mock_handle.assert_called_once()


def test_main_unknown_command():
  """Verifies the behavior of main with an unknown command (if argparser doesn't catch it)."""
  # To test the final return 0 fallback, we need to bypass argparse validation
  import argparse

  with patch.object(argparse.ArgumentParser, "parse_args") as mock_parse:

    class DummyArgs:
      """A dummy arguments class for mocking."""

      command = "unknown_cmd"
      verbose = False
      log_file = None
      no_color = False

    mock_parse.return_value = DummyArgs()
    assert main(["unknown_cmd"]) == 0


def test_main_cli_execution():
  """Tests running the module directly"""
  import runpy
  import sys

  with patch("sys.exit") as mock_exit:
    with patch.object(sys, "argv", ["ml_switcheroo", "schema"]):
      # run_path will parse the file, execute it in __main__ namespace
      # which calls main() -> handle_schema() -> prints schema -> sys.exit(0)
      runpy.run_path("src/ml_switcheroo/cli/__main__.py", run_name="__main__")
      mock_exit.assert_called_once_with(0)
