"""
Tests for 'define --dry-run' command.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from ml_switcheroo.cli.handlers.define import handle_define


@pytest.fixture
def dry_run_yaml(tmp_path):
  """Create valid YAML for inputs."""
  f = tmp_path / "dry.yaml"
  f.write_text("""
operation: "DryOp"
description: "TestOp"
std_args: []
variants:
  torch:
    api: "torch.dry"
scaffold_plugins:
  - name: "dry_plugin"
    type: "call_transform"
    doc: "dry"
""")
  return f


def test_dry_run_no_writes(dry_run_yaml, capsys, tmp_path):
  """
  Scenario: Run define with dry_run=True.
  Expectation: Files are read, Diffs printed log style, but Write is never called.
  """
  # Prepare dummy files
  standards_file = tmp_path / "standards_internal.py"
  standards_file.write_text("INTERNAL_OPS = {}", encoding="utf-8")

  torch_json = tmp_path / "torch.json"
  # Ensure it doesn't exist to prove dry run doesn't create it,
  # OR create it empty to prove it doesn't modify.
  # Let's mock the path resolution to point to a non-existent file.
  json_target = tmp_path / "definitions" / "torch.json"

  # Patches
  p1 = patch("inspect.getfile", return_value=str(standards_file))
  p2 = patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=json_target)
  p3 = patch("ml_switcheroo.cli.handlers.define.get_adapter", return_value=MagicMock())

  with p1, p2, p3:
    if "yaml" not in sys.modules:
      m_yaml = MagicMock()
      # Return list structure of ops
      m_yaml.safe_load.return_value = [
        {
          "operation": "DryOp",
          "description": "TestOp",
          "std_args": [],
          "variants": {"torch": {"api": "torch.dry"}},
          "scaffold_plugins": [{"name": "p", "type": "call_transform"}],
        }
      ]
      with patch("ml_switcheroo.cli.handlers.define.yaml", m_yaml):
        handle_define(dry_run_yaml, dry_run=True)
    else:
      handle_define(dry_run_yaml, dry_run=True)

  # Checks
  outerr = capsys.readouterr()
  stdout = outerr.out

  # 1. Verify JSON Dry Run Output (FrameworkInjector)
  assert "[Dry Run] Writing to torch.json" in stdout
  assert '"api": "torch.dry"' in stdout

  # 2. Verify Hub Dry Run Output (StandardsInjector logs)
  # The new define handler prints "[Dry Run] Would update standards_internal.py"
  assert "Would update standards_internal.py" in stdout

  # 3. Verify Files Unchanged/Uncreated
  assert not json_target.exists()
  assert standards_file.read_text("utf-8") == "INTERNAL_OPS = {}"
