from ml_switcheroo.cli.handlers.define import handle_define
import pytest


def test_define_dry_run(tmp_path):
  f = tmp_path / "def.yaml"
  f.write_text("name: Abs\noperation: abs\ndescription: ''\nvariants: {}\ntier: MATH\nstd_args: [x]")

  assert handle_define(f, dry_run=True, no_test_gen=True) == 0
