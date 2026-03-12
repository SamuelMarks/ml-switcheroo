from ml_switcheroo.cli.handlers.define import handle_define
import pytest
from pathlib import Path


def test_define_file_not_found(tmp_path):
  f = tmp_path / "def.yaml"
  assert handle_define(f) == 1


def test_define_yaml_list(tmp_path):
  f = tmp_path / "def.yaml"
  f.write_text(
    "- name: Abs\n  operation: abs\n  description: ''\n  variants: {torch: {name: torch.abs}}\n  tier: MATH\n  std_args: [x]"
  )
  assert handle_define(f, dry_run=True, no_test_gen=True) == 0


def test_define_empty(tmp_path):
  f = tmp_path / "def.yaml"
  f.write_text("")
  assert handle_define(f, dry_run=True, no_test_gen=True) == 0
