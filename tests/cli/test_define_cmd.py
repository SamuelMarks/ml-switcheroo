from ml_switcheroo.cli.handlers.define import handle_define
import pytest


def test_define_cmd_stdin(capsys):
  import sys
  from unittest.mock import patch
  import io

  yaml_data = "name: Abs\noperation: abs\ndescription: ''\nvariants: {torch: {name: torch.abs, requires_plugin: 'AbsPlugin'}}\ntier: MATH\nstd_args: [x]"

  with patch("sys.stdin", io.StringIO(yaml_data)):
    from pathlib import Path

    f = Path("-")
    assert handle_define(f, dry_run=True, no_test_gen=True) == 0
