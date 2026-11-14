"""Tests for `scripts/ensure_ops.py` using real temporary directories."""

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import scripts.ensure_ops


def test_run_creates_missing_yaml_files(tmp_path: Path) -> None:
  """Verify `run()` creates missing YAML files without overwriting existing ones.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  existing_file = odl_dir / "Conv1d.yaml"
  existing_file.write_text("operation: CustomConv1d\n", encoding="utf-8")

  created = scripts.ensure_ops.run(odl_dir=odl_dir)
  assert created == len(scripts.ensure_ops.BASELINE_OPS) - 1

  # Existing file must not be overwritten
  assert yaml.safe_load(existing_file.read_text(encoding="utf-8"))["operation"] == "CustomConv1d"

  # Newly created file must match schema structure
  conv2d_data = yaml.safe_load((odl_dir / "Conv2d.yaml").read_text(encoding="utf-8"))
  assert conv2d_data["operation"] == "Conv2d"
  assert conv2d_data["description"] == "Verified API: Conv2d"
  assert conv2d_data["std_args"] == []
  assert conv2d_data["variants"] == {}


def test_main_and_dunder_main_execution(tmp_path: Path) -> None:
  """Verify `main()` and `__main__` entrypoint execution on a temporary ODL directory.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl_cli"
  odl_dir.mkdir()

  with patch.object(sys, "argv", ["scripts/ensure_ops.py", "--odl-dir", str(odl_dir)]):
    with pytest.raises(SystemExit) as exc:
      runpy.run_module("scripts.ensure_ops", run_name="__main__")
    assert exc.value.code == 0

  assert (odl_dir / "Linear.yaml").exists()
