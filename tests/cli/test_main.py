"""Test module."""

import argparse
from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.cli.__main__ import main


@patch("ml_switcheroo.cli.__main__.commands.handle_convert")
def test_main_convert(mock_handle_convert):
  """Test element."""
  mock_handle_convert.return_value = 0
  argv = ["convert", "input.py", "--target", "jax", "--out", "out.py", "--config", "a=1"]
  res = main(argv)
  assert res == 0
  mock_handle_convert.assert_called_once()
  # verify args
  call_args = mock_handle_convert.call_args[0]
  assert call_args[0] == Path("input.py")
  assert call_args[1] == Path("out.py")
  assert call_args[3] == "jax"
  assert call_args[7] == {"a": 1}  # settings parsed via parse_cli_key_values converts '1' to int 1


@patch("ml_switcheroo.cli.__main__.commands.handle_define")
def test_main_define(mock_handle_define):
  """Test element."""
  mock_handle_define.return_value = 0
  argv = ["define", "op.yaml"]
  res = main(argv)
  assert res == 0
  mock_handle_define.assert_called_once_with(Path("op.yaml"))


@patch("ml_switcheroo.cli.__main__.commands.handle_gen_weight_script")
def test_main_gen_weight_script(mock_handle):
  """Test element."""
  mock_handle.return_value = 0
  argv = ["gen-weight-script", "model.py", "--out", "script.py", "--source", "torch", "--target", "jax"]
  res = main(argv)
  assert res == 0
  mock_handle.assert_called_once_with(Path("model.py"), Path("script.py"), "torch", "jax")


@patch("ml_switcheroo.cli.__main__.commands.handle_matrix")
def test_main_matrix(mock_handle):
  """Test element."""
  mock_handle.return_value = 0
  res = main(["matrix"])
  assert res == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.handle_schema")
def test_main_schema(mock_handle):
  """Test element."""
  mock_handle.return_value = 0
  res = main(["schema"])
  assert res == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.handle_suggest")
def test_main_suggest(mock_handle):
  """Test element."""
  mock_handle.return_value = 0
  res = main(["suggest", "torch.add", "--out-dir", "out"])
  assert res == 0
  mock_handle.assert_called_once_with("torch.add", out_dir=Path("out"), batch_size=50)


@patch("ml_switcheroo.cli.__main__.handle_scaffold")
def test_main_scaffold(mock_handle):
  """Test element."""
  res = main(["scaffold", "jax"])
  assert res == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.handle_harvest")
def test_main_harvest(mock_handle):
  """Test element."""
  res = main(["harvest", "tests/"])
  assert res == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_ci")
def test_main_ci(mock_handle):
  """Test element."""
  mock_handle.return_value = 0
  res = main(["ci", "--repair"])
  assert res == 0
  mock_handle.assert_called_once_with(False, Path("README.md"), None, True)


@patch("ml_switcheroo.cli.__main__.commands.handle_docs")
def test_main_gen_docs(mock_handle):
  """Test element."""
  mock_handle.return_value = 0
  res = main(["gen-docs"])
  assert res == 0
  mock_handle.assert_called_once_with("torch", "jax", Path("MIGRATION_GUIDE.md"))


@patch("ml_switcheroo.cli.__main__.commands.handle_gen_tests")
def test_main_gen_tests(mock_handle):
  """Test element."""
  mock_handle.return_value = 0
  res = main(["gen-tests"])
  assert res == 0
  mock_handle.assert_called_once()


@patch("builtins.open")
@patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline")
def test_main_verified_pipeline(mock_run, mock_open):
  """Test element."""
  mock_run.return_value = {"status": "success"}
  mock_open.return_value.__enter__.return_value.read.return_value = "code"
  res = main(["verified-pipeline", "file.py"])
  assert res == 0
  mock_run.assert_called_once_with("code")


@patch("builtins.open")
@patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline")
def test_main_verified_pipeline_fail(mock_run, mock_open):
  """Test element."""
  mock_run.return_value = {"status": "error"}
  mock_open.return_value.__enter__.return_value.read.return_value = "code"
  res = main(["verified-pipeline", "file.py"])
  assert res == 1


def test_main_unknown():
  """Test element."""
  # Simulate an unknown command (argparse would usually catch this, but just in case)
  with patch("argparse.ArgumentParser.parse_args") as mock_parse:
    mock_parse.return_value = argparse.Namespace(command="unknown")
    res = main(["unknown"])
    assert res == 0
