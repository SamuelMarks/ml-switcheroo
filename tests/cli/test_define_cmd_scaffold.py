from ml_switcheroo.cli.handlers.define import handle_define
import pytest


def test_define_cmd_scaffold(tmp_path):
  f = tmp_path / "def.yaml"
  f.write_text(
    "name: Abs\noperation: abs\ndescription: ''\nvariants: {torch: {name: torch.abs, requires_plugin: 'AbsPlugin'}}\ntier: MATH\nstd_args: [x]"
  )

  assert handle_define(f, dry_run=True, no_test_gen=True) == 0
