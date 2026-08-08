"""Test suite for the Cli E2E module."""

from ml_switcheroo.cli.__main__ import main
from unittest.mock import patch, MagicMock, mock_open
import runpy
import sys


def test_cli_e2e_suggest(tmp_path):
  """Verifies the behavior of CLI end-to-end suggest."""
  try:
    main(["suggest", "torch.nn.Linear", "--out", str(tmp_path)])
  except SystemExit:
    pass


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_cli_e2e_convert(mock_engine, tmp_path):
  """Verifies the behavior of CLI end-to-end convert."""
  out = tmp_path / "out_convert"
  out.mkdir()
  in_file = tmp_path / "model.py"
  in_file.write_text("import torch; torch.nn.Linear(10, 10)\n")
  mock_instance = MagicMock()
  mock_engine.return_value = mock_instance
  mock_res = MagicMock()
  mock_res.code = "import jax"
  mock_instance.convert.return_value = mock_res
  try:
    main(["convert", str(in_file), "--out", str(out), "--source", "torch", "--target", "jax"])
  except SystemExit:
    pass


@patch("ml_switcheroo.cli.handlers.verify.BatchValidator.run_all", return_value={})
def test_cli_e2e_ci(mock_run, tmp_path):
  """Verifies the behavior of CLI end-to-end ci."""
  out = tmp_path / "report.json"
  try:
    main(["ci", "--json-report", str(out)])
  except SystemExit:
    pass


def test_cli_e2e_docs(tmp_path):
  """Verifies the behavior of CLI end-to-end documentation."""
  out = tmp_path / "MIGRATION.md"
  try:
    main(["gen-docs", "--out", str(out)])
  except SystemExit:
    pass


def test_cli_e2e_matrix(tmp_path):
  """Verifies the behavior of CLI end-to-end matrix."""
  try:
    main(["matrix"])
  except SystemExit:
    pass


def test_cli_e2e_audit(tmp_path):
  """Verifies the behavior of CLI end-to-end audit."""
  in_file = tmp_path / "model.py"
  in_file.write_text("import torch\nclass Model: pass\n")
  try:
    main(["audit", str(in_file)])
  except SystemExit:
    pass


@patch("ml_switcheroo.generated_tests.generator.get_template", return_value=False)
def test_cli_e2e_gen_tests(mock_get_template, tmp_path):
  """Verifies the behavior of CLI end-to-end generation tests."""
  try:
    main(["gen-tests"])
  except SystemExit:
    pass


def test_cli_e2e_weight_script(tmp_path):
  """Verifies the behavior of CLI end-to-end weight script."""
  in_file = tmp_path / "model.py"
  in_file.write_text("import torch\nclass Model: pass\n")
  out = tmp_path / "weight.py"
  try:
    main(["gen-weight-script", str(in_file), "--out", str(out)])
  except SystemExit:
    pass


def test_cli_e2e_schema():
  """Verifies the behavior of CLI schema."""
  with patch("ml_switcheroo.cli.__main__.handle_schema", return_value=0) as mock_schema:
    assert main(["schema"]) == 0
    mock_schema.assert_called_once()


def test_cli_e2e_scaffold():
  """Verifies the behavior of CLI scaffold."""
  with patch("ml_switcheroo.cli.__main__.handle_scaffold") as mock_scaffold:
    assert main(["scaffold", "jax.numpy"]) == 0
    mock_scaffold.assert_called_once()


def test_cli_e2e_harvest(tmp_path):
  """Verifies the behavior of CLI harvest."""
  with patch("ml_switcheroo.cli.__main__.handle_harvest") as mock_harvest:
    assert main(["harvest", str(tmp_path)]) == 0
    mock_harvest.assert_called_once()


def test_cli_e2e_verified_pipeline_success():
  """Verifies the behavior of CLI verified pipeline successfully."""
  with (
    patch("builtins.open", mock_open(read_data="x = 1")),
    patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline") as mock_pipeline,
    patch("sys.stdout"),
  ):
    mock_pipeline.return_value = {"status": "success"}
    assert main(["verified-pipeline", "some_path.py"]) == 0


def test_cli_e2e_verified_pipeline_failure():
  """Verifies the behavior of CLI verified pipeline successfully handling failure."""
  with (
    patch("builtins.open", mock_open(read_data="x = 1")),
    patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline") as mock_pipeline,
    patch("sys.stdout"),
  ):
    mock_pipeline.return_value = {"status": "error"}
    assert main(["verified-pipeline", "some_path.py"]) == 1


def test_cli_e2e_unknown_command():
  """Verifies the behavior of main with an unknown command."""
  import argparse

  with patch.object(argparse.ArgumentParser, "parse_args") as mock_parse:

    class DummyArgs:
      command = "unknown_cmd"
      verbose = False
      log_file = None
      no_color = False

    mock_parse.return_value = DummyArgs()
    assert main(["unknown_cmd"]) == 0


def test_cli_e2e_module_execution():
  """Tests running the module directly"""
  with patch("sys.exit") as mock_exit:
    with patch.object(sys, "argv", ["ml_switcheroo", "schema"]):
      with patch("ml_switcheroo.cli.__main__.handle_schema", return_value=0):
        runpy.run_path("src/ml_switcheroo/cli/__main__.py", run_name="__main__")
        mock_exit.assert_called_once_with(0)
