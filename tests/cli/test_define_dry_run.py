"""
Tests for 'define --dry-run' command.

Verifies that:
1.  Source files are NOT modified on disk.
2.  Diff output is printed to stdout.
3.  Plugin generation logic is skipped (logged).
"""

import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from ml_switcheroo.cli.handlers.define import handle_define


@pytest.fixture
def dry_run_yaml(tmp_path):
  """Create valid YAML for inputs."""
  f = tmp_path / "dry.yaml"
  f.write_text(
    """ 
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
"""
  )
  return f


@pytest.fixture
def mock_codebase():
  """
  Mocks the source code content for Hub and Spoke.
  """
  hub_code = "INTERNAL_OPS = {}\n"
  spoke_code = """ 
@register_framework("torch") 
class TorchAdapter: 
    @property
    def definitions(self): 
        return {} 
"""
  return hub_code, spoke_code


def test_dry_run_no_writes(dry_run_yaml, mock_codebase, capsys):
  """
  Scenario: Run define with dry_run=True.
  Expectation: Files are read, Diffs printed, but Write is never called.
  """
  hub_code, spoke_code = mock_codebase

  # --- Real File Strategy ---
  # Create a fake source tree in tmp_path
  src_dir = dry_run_yaml.parent / "src"
  src_dir.mkdir()

  real_hub = src_dir / "standards.py"
  real_hub.write_text(hub_code, encoding="utf-8")

  real_spoke = src_dir / "torch.py"
  real_spoke.write_text(spoke_code, encoding="utf-8")

  # Patch inspect to point to these real files
  with patch("inspect.getfile") as mock_getfile:

    def file_map(obj):
      n = getattr(obj, "__name__", "")
      if "standards" in n:
        return str(real_hub)
      if "TorchAdapter" in n:
        return str(real_spoke)
      return __file__

    mock_getfile.side_effect = file_map

    # Mock Adapter
    mock_adp = MagicMock()
    type(mock_adp).__name__ = "TorchAdapter"
    with patch("ml_switcheroo.cli.handlers.define.get_adapter", return_value=mock_adp):
      # Mock dependencies
      if "yaml" not in sys.modules:
        m_yaml = MagicMock()
        m_yaml.safe_load.return_value = [
          {
            "operation": "DryOp",
            "description": "Desc",
            "std_args": [],
            "variants": {"torch": {"api": "new.api"}},
            "scaffold_plugins": [{"name": "p", "type": "call_transform", "doc": "d"}],
          }
        ]
        with patch("ml_switcheroo.cli.handlers.define.yaml", m_yaml):
          handle_define(dry_run_yaml, dry_run=True)
      else:
        handle_define(dry_run_yaml, dry_run=True)

  # Checks
  outerr = capsys.readouterr()
  stdout = outerr.out

  # 1. Verify Diffs Printed
  assert "--- Changes for" in stdout
  # Relaxed check for content presence to handle potential formatting diffs
  # LibCST generated dict might be inline or multiline
  assert "DryOp" in stdout
  assert "StandardMap" in stdout

  # 2. Verify Plugins Skipped
  assert "[Dry Run] Would generate plugin file" in stdout

  # 3. Verify Tests Skipped
  assert "[Dry Run] Would generate test file" in stdout

  # 4. Verify Files Unchanged on Disk
  assert real_hub.read_text("utf-8") == hub_code
  assert real_spoke.read_text("utf-8") == spoke_code
