"""Test suite for the CLI end-to-end command execution."""

import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from ml_switcheroo.cli.__main__ import main


def test_cli_e2e_suggest(tmp_path: Path) -> None:
  """Verify CLI end-to-end suggest command returns exit code 0 and writes suggested YAML.

  Args:
      tmp_path (Path): Temporary directory pytest fixture.
  """
  res = main(["suggest", "math.sqrt", "--out", str(tmp_path)])
  assert res == 0
  assert any(tmp_path.iterdir())


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_cli_e2e_convert(mock_engine: MagicMock, tmp_path: Path) -> None:
  """Verify CLI end-to-end convert command writes output and returns 0.

  Args:
      mock_engine (MagicMock): Mocked ASTEngine class.
      tmp_path (Path): Temporary directory pytest fixture.
  """
  out: Path = tmp_path / "out_convert.py"
  in_file: Path = tmp_path / "model.py"
  in_file.write_text("import torch; torch.nn.Linear(10, 10)\n")
  mock_instance: MagicMock = MagicMock()
  mock_engine.return_value = mock_instance
  mock_res: MagicMock = MagicMock()
  mock_res.code = "import jax"
  mock_res.success = True
  mock_res.errors = []
  mock_instance.run.return_value = mock_res
  res = main(["convert", str(in_file), "--out", str(out), "--source", "torch", "--target", "jax"])
  assert res == 0
  assert out.exists()
  assert out.read_text() == "import jax"


@patch("ml_switcheroo.cli.handlers.verify.BatchValidator.run_all", return_value={})
def test_cli_e2e_ci(mock_run: MagicMock, tmp_path: Path) -> None:
  """Verify CLI end-to-end ci command generates a JSON report and returns 0.

  Args:
      mock_run (MagicMock): Mocked BatchValidator.run_all method.
      tmp_path (Path): Temporary directory pytest fixture.
  """
  out: Path = tmp_path / "report.json"
  res = main(["ci", "--json-report", str(out)])
  assert res == 0
  assert out.exists()
  assert out.read_text().strip() == "{}"


def test_cli_e2e_docs(tmp_path: Path) -> None:
  """Verify CLI end-to-end gen-docs command writes migration markdown and returns 0.

  Args:
      tmp_path (Path): Temporary directory pytest fixture.
  """
  out: Path = tmp_path / "MIGRATION.md"
  res = main(["gen-docs", "--out", str(out)])
  assert res == 0
  assert out.exists()
  assert len(out.read_text()) > 0


def test_cli_e2e_matrix(tmp_path: Path) -> None:
  """Verify CLI end-to-end matrix command renders compatibility table and returns 0.

  Args:
      tmp_path (Path): Temporary directory pytest fixture.
  """
  res = main(["matrix"])
  assert res == 0


def test_cli_e2e_audit(tmp_path: Path) -> None:
  """Verify CLI end-to-end rejects invalid 'audit' subcommand with SystemExit(2).

  Args:
      tmp_path (Path): Temporary directory pytest fixture.
  """
  import pytest

  in_file: Path = tmp_path / "model.py"
  in_file.write_text("import torch\nx = torch.abs(y)\n")
  with pytest.raises(SystemExit) as exc_info:
    main(["audit", str(in_file)])
  assert exc_info.value.code == 2


@patch("ml_switcheroo.generated_tests.generator.get_template", return_value=False)
def test_cli_e2e_gen_tests(mock_get_template: MagicMock, tmp_path: Path) -> None:
  """Verify CLI end-to-end gen-tests command executes, writes output file, and returns 0.

  Args:
      mock_get_template (MagicMock): Mocked get_template helper.
      tmp_path (Path): Temporary directory pytest fixture.
  """
  out_file: Path = tmp_path / "gen_test.py"
  res = main(["gen-tests", "--out", str(out_file)])
  assert res == 0
  assert out_file.exists()


def test_cli_e2e_weight_script(tmp_path: Path) -> None:
  """Verify CLI end-to-end gen-weight-script command writes migration script and returns 0.

  Args:
      tmp_path (Path): Temporary directory pytest fixture.
  """
  in_file: Path = tmp_path / "model.py"
  in_file.write_text(
    "import torch.nn as nn\n"
    "class Model(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc = nn.Linear(4, 4)\n"
  )
  out: Path = tmp_path / "weight.py"
  res = main(["gen-weight-script", str(in_file), "--out", str(out), "--source", "torch", "--target", "flax_nnx"])
  assert res == 0
  assert out.exists()
  assert len(out.read_text()) > 0


def test_cli_e2e_schema() -> None:
  """Verifies the behavior of CLI schema."""
  with patch("ml_switcheroo.cli.__main__.handle_schema", return_value=0) as mock_schema:
    assert main(["schema"]) == 0
    mock_schema.assert_called_once()


def test_cli_e2e_scaffold() -> None:
  """Verifies the behavior of CLI scaffold."""
  with patch("ml_switcheroo.cli.__main__.handle_scaffold") as mock_scaffold:
    assert main(["scaffold", "jax.numpy"]) == 0
    mock_scaffold.assert_called_once()


def test_cli_e2e_harvest(tmp_path: Path) -> None:
  """Verifies the behavior of CLI harvest.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  with patch("ml_switcheroo.cli.__main__.handle_harvest") as mock_harvest:
    assert main(["harvest", str(tmp_path)]) == 0
    mock_harvest.assert_called_once()


def test_cli_e2e_verified_pipeline_success() -> None:
  """Verifies the behavior of CLI verified pipeline successfully."""
  with (
    patch("builtins.open", mock_open(read_data="x = 1")),
    patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline") as mock_pipeline,
    patch("sys.stdout"),
  ):
    mock_pipeline.return_value = {"status": "success"}
    assert main(["verified-pipeline", "some_path.py"]) == 0


def test_cli_e2e_verified_pipeline_failure() -> None:
  """Verifies the behavior of CLI verified pipeline successfully handling failure."""
  with (
    patch("builtins.open", mock_open(read_data="x = 1")),
    patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline") as mock_pipeline,
    patch("sys.stdout"),
  ):
    mock_pipeline.return_value = {"status": "error"}
    assert main(["verified-pipeline", "some_path.py"]) == 1


def test_cli_e2e_unknown_command() -> None:
  """Verifies the behavior of main with an unknown command."""
  import argparse

  with patch.object(argparse.ArgumentParser, "parse_args") as mock_parse:

    class DummyArgs:
      """Test helper class DummyArgs."""

      command: str = "unknown_cmd"
      verbose: bool = False
      log_file = None
      no_color: bool = False

    mock_parse.return_value = DummyArgs()
    assert main(["unknown_cmd"]) == 0


def test_cli_e2e_module_execution() -> None:
  """Verify cli e2e module execution behavior."""
  with patch("sys.exit") as mock_exit:
    with patch.object(sys, "argv", ["ml_switcheroo", "schema"]):
      with patch("ml_switcheroo.cli.__main__.handle_schema", return_value=0):
        runpy.run_path("src/ml_switcheroo/cli/__main__.py", run_name="__main__")
        mock_exit.assert_called_once_with(0)
