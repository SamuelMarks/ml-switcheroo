"""
Tests for define --dry-run.
"""

import sys
from unittest.mock import patch, MagicMock
from pathlib import Path
from ml_switcheroo.cli.handlers.define import handle_define


def test_dry_run_no_writes_(tmp_path, capsys):
  """
  Scenario: Run define with dry_run=True.
  Expectation: Prints '[Dry Run] Writing to...' for JSONs.
  """
  yaml_file = tmp_path / "op.yaml"
  yaml_file.write_text(
    """
operation: "DryOp"
description: "TestOp"
std_args: []
variants:
  torch: { api: "torch.dry" }
""",
    encoding="utf-8",
  )

  # Patch path resolvers to use tmp_path
  spec_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "definitions"

  # Patches
  p1 = patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir", return_value=spec_dir)
  p2 = patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", side_effect=lambda fw: snap_dir / f"{fw}.json")
  p3 = patch("ml_switcheroo.cli.handlers.define.get_adapter", return_value=MagicMock())

  with p1, p2, p3:
    handle_define(yaml_file, dry_run=True, no_test_gen=True)

  captured = capsys.readouterr()
  stdout = captured.out

  # Check Hub (Spec)
  assert (
    "[Dry Run] Writing to k_array_api.json" in stdout
    or "[Dry Run] Writing to k_neural_net.json" in stdout
    or "[Dry Run] Writing to k_framework_extras.json" in stdout
  )

  # Check Spoke (Map)
  assert "[Dry Run] Writing to torch.json" in stdout
  assert '"api": "torch.dry"' in stdout
