"""Tests for `scripts/checkoff_ops.py` using real temporary markdown files."""

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.checkoff_ops


def test_run_checks_off_all_or_specific_ops(tmp_path: Path) -> None:
  """Verify `run()` checks off either specific ops or all backticked dotted ops.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  plan_file = tmp_path / "TODO_PLAN.md"
  initial = "# TODO PLAN\n- [ ] `some_dialect.some_op`\n- [ ] `another.op`\n- [x] `already.done`\n"
  plan_file.write_text(initial, encoding="utf-8")

  # Check off only specific op first
  count_single = scripts.checkoff_ops.run(plan_file=plan_file, ops=["some_dialect.some_op"])
  assert count_single == 1
  content_after_single = plan_file.read_text(encoding="utf-8")
  assert "- [x] `some_dialect.some_op`" in content_after_single
  assert "- [ ] `another.op`" in content_after_single

  # Check off remaining ops
  count_all = scripts.checkoff_ops.run(plan_file=plan_file)
  assert count_all == 1
  expected = "# TODO PLAN\n- [x] `some_dialect.some_op`\n- [x] `another.op`\n- [x] `already.done`\n"
  assert plan_file.read_text(encoding="utf-8") == expected


def test_main_and_dunder_main_execution(tmp_path: Path) -> None:
  """Verify `checkoff_ops.py` CLI execution via `runpy.run_module`.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  plan_file = tmp_path / "PLAN.md"
  plan_file.write_text("- [ ] `torch.add`\n", encoding="utf-8")

  with patch.object(sys, "argv", ["scripts/checkoff_ops.py", "--plan-file", str(plan_file)]):
    with pytest.raises(SystemExit) as exc:
      runpy.run_module("scripts.checkoff_ops", run_name="__main__")
    assert exc.value.code == 0

  assert plan_file.read_text(encoding="utf-8") == "- [x] `torch.add`\n"
