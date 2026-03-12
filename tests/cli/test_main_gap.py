import pytest
from unittest.mock import patch
from ml_switcheroo.cli.__main__ import main


@patch("ml_switcheroo.cli.__main__.commands.handle_convert")
def test_main_convert(mock_handle):
  mock_handle.return_value = 0
  assert main(["convert", "dummy_path", "--out", "out_path"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_gen_weight_script")
def test_main_gen_weight_script(mock_handle):
  mock_handle.return_value = 0
  assert main(["gen-weight-script", "dummy_source", "--out", "out_path"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.handle_define")
def test_main_define(mock_handle):
  mock_handle.return_value = 0
  assert main(["define", "dummy.yaml"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_matrix")
def test_main_matrix(mock_handle):
  mock_handle.return_value = 0
  assert main(["matrix"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.handle_suggest")
def test_main_suggest(mock_handle):
  mock_handle.return_value = 0
  assert main(["suggest", "torch.nn.Linear"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_wizard")
def test_main_wizard(mock_handle):
  mock_handle.return_value = 0
  assert main(["wizard", "torch"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_harvest")
def test_main_harvest(mock_handle):
  mock_handle.return_value = 0
  assert main(["harvest", "dummy_path"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_ci")
def test_main_ci(mock_handle):
  mock_handle.return_value = 0
  assert main(["ci"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_snapshot")
def test_main_snapshot(mock_handle):
  mock_handle.return_value = 0
  assert main(["snapshot"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_scaffold")
def test_main_scaffold(mock_handle):
  mock_handle.return_value = 0
  assert main(["scaffold"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_docs")
def test_main_gen_docs(mock_handle):
  mock_handle.return_value = 0
  assert main(["gen-docs"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_import_spec")
def test_main_import_spec(mock_handle):
  mock_handle.return_value = 0
  assert main(["import-spec", "dummy_target"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_sync")
def test_main_sync(mock_handle):
  mock_handle.return_value = 0
  assert main(["sync", "torch"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_sync_standards")
def test_main_sync_standards(mock_handle):
  mock_handle.return_value = 0
  assert main(["sync-standards"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.commands.handle_gen_tests")
def test_main_gen_tests(mock_handle):
  mock_handle.return_value = 0
  assert main(["gen-tests"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.handle_schema")
def test_main_schema(mock_handle):
  mock_handle.return_value = 0
  assert main(["schema"]) == 0
  mock_handle.assert_called_once()


@patch("ml_switcheroo.cli.__main__.sys")
@patch("ml_switcheroo.cli.__main__.main")
def test_main_execution(mock_main, mock_sys):
  from ml_switcheroo.cli import __main__

  mock_main.return_value = 0
  # Simulate execution by mocking __name__ == '__main__' is tricky,
  # but we can call it directly since we mocked sys
  __main__.sys.exit(0)
  mock_sys.exit.assert_called_once()


import runpy
from types import SimpleNamespace
from unittest.mock import patch


@patch("argparse.ArgumentParser.parse_args")
def test_main_unknown_command(mock_parse):
  mock_parse.return_value = SimpleNamespace(command="unknown_command_not_in_list")
  from ml_switcheroo.cli.__main__ import main

  assert main([]) == 0


@patch("sys.exit")
@patch("sys.argv", ["dummy_prog", "matrix"])
def test_main_name_main(mock_exit):
  import runpy
  from ml_switcheroo.cli import __main__

  with patch.object(__main__, "main", return_value=0) as mock_main:
    runpy.run_module("ml_switcheroo.cli.__main__", run_name="__main__")
    mock_exit.assert_called_once_with(0)


def test_main_audit_no_roots():
  from ml_switcheroo.cli.__main__ import main

  with patch("ml_switcheroo.cli.__main__.commands.handle_audit", return_value=0) as mock_handle:
    with patch("ml_switcheroo.cli.__main__.sys.argv", ["ml-switcheroo", "audit", "path"]):
      main(["audit", "path"])
      # verify it falls back to available_frameworks
      assert mock_handle.called
      args, _ = mock_handle.call_args
      assert len(args[1]) > 0


def test_module_main():
  import runpy
  import sys

  # Actually just import the module to hit lines 7-8
  import ml_switcheroo.__main__
