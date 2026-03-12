import sys
import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.cli.__main__ import main


@pytest.mark.parametrize(
  "command_args,mock_target",
  [
    (["audit", "some/path"], "ml_switcheroo.cli.commands.handle_audit"),
    (["audit", "some/path", "--roots", "torch", "jax"], "ml_switcheroo.cli.commands.handle_audit"),
    (
      ["convert", "in.py", "--out", "out.py", "--source", "torch", "--target", "jax"],
      "ml_switcheroo.cli.commands.handle_convert",
    ),
    (
      ["gen-weight-script", "in.py", "--out", "out.py", "--source", "torch", "--target", "jax"],
      "ml_switcheroo.cli.commands.handle_gen_weight_script",
    ),
    (["define", "test.yaml"], "ml_switcheroo.cli.__main__.handle_define"),
    (["matrix"], "ml_switcheroo.cli.commands.handle_matrix"),
    (["schema"], "ml_switcheroo.cli.__main__.handle_schema"),
    (["suggest", "api"], "ml_switcheroo.cli.__main__.handle_suggest"),
    (["wizard", "torch"], "ml_switcheroo.cli.commands.handle_wizard"),
    (["harvest", "path", "--target", "torch"], "ml_switcheroo.cli.commands.handle_harvest"),
    (["ci"], "ml_switcheroo.cli.commands.handle_ci"),
    (["snapshot"], "ml_switcheroo.cli.commands.handle_snapshot"),
    (["scaffold"], "ml_switcheroo.cli.commands.handle_scaffold"),
    (["gen-docs"], "ml_switcheroo.cli.commands.handle_docs"),
    (["import-spec", "torch"], "ml_switcheroo.cli.commands.handle_import_spec"),
    (["sync", "torch"], "ml_switcheroo.cli.commands.handle_sync"),
    (["sync-standards"], "ml_switcheroo.cli.commands.handle_sync_standards"),
    (["gen-tests", "--out", "out.py"], "ml_switcheroo.cli.commands.handle_gen_tests"),
  ],
)
def test_main_routing(command_args, mock_target):
  # patch path existence check or handle_define so it doesn't fail on missing file
  if command_args[0] == "define":
    with patch("ml_switcheroo.cli.__main__.handle_define") as mock_cmd:
      mock_cmd.return_value = 0
      main(command_args)
      mock_cmd.assert_called_once()
  else:
    with patch(mock_target) as mock_cmd:
      mock_cmd.return_value = 0
      main(command_args)
      mock_cmd.assert_called_once()


def test_main_execution_cli_dunder_main():
  with patch("sys.argv", ["ml-switcheroo", "--help"]):
    with patch.object(sys, "exit") as mock_exit:
      with patch("ml_switcheroo.cli.__main__.__name__", "__main__"):
        import runpy

        try:
          runpy.run_module("ml_switcheroo.cli.__main__", run_name="__main__")
        except SystemExit:
          pass
        mock_exit.assert_called()
