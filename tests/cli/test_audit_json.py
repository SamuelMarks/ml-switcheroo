"""
Tests for Audit Command JSON Mode.

Verifies:
1. `audit --json` outputs valid JSON.
2. Output structure contains expected fields.
3. Console logging is suppressed in JSON mode.
4. CLI integration works correctly.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from ml_switcheroo.cli.handlers.audit import handle_audit
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.cli.__main__ import main


class MockAuditSemantics(SemanticsManager):
  """
  Mock Semantics Manager for Audit tests.
  Bypasses file loading and provides a deterministic lookup dictionary.
  """

  def __init__(self, known_apis: dict):
    # Bypass super init to avoid file I/O
    self.known = known_apis

  def get_definition(self, api_name: str):
    if api_name in self.known:
      fw_key = self.known[api_name]
      return "op_id", {"variants": {fw_key: {"api": api_name}}}
    return None


def test_audit_json_output_structure(tmp_path, capsys):
  """
  Scenario: Mixed supported and unsupported operations.
  Expectation: Valid JSON list with correct fields.
  """
  # 1. Setup Source File
  code = """ 
import torch
x = torch.abs(y) 
z = torch.mystery(y) 
"""
  f = tmp_path / "model.py"
  f.write_text(code, encoding="utf-8")

  # 2. Setup Mock Semantics
  # torch.abs is known, torch.mystery is unknown
  mock_mgr = MockAuditSemantics({"torch.abs": "torch"})

  # 3. Run Handler with json_mode=True
  with patch("ml_switcheroo.cli.handlers.audit.SemanticsManager", return_value=mock_mgr):
    # Mock log_info to verify it's NOT called
    with patch("ml_switcheroo.cli.handlers.audit.log_info") as mock_log:
      ret = handle_audit(f, ["torch"], json_mode=True)

      # Verify logs suppressed
      mock_log.assert_not_called()

  # 4. Verify Output
  captured = capsys.readouterr()
  output_json = captured.out.strip()

  assert output_json.startswith("[")
  assert output_json.endswith("]")

  data = json.loads(output_json)
  assert isinstance(data, list)
  assert len(data) == 2

  # Sort to ensure stable indexing
  data.sort(key=lambda x: x["api"])

  # Item 1: torch.abs (Supported)
  item1 = data[0]
  assert item1["api"] == "torch.abs"
  assert item1["supported"] is True
  assert item1["framework"] == "torch"
  assert "suggestion" not in item1

  # Item 2: torch.mystery (Unsupported)
  item2 = data[1]
  assert item2["api"] == "torch.mystery"
  assert item2["supported"] is False
  assert item2["framework"] == "torch"
  assert "suggestion" in item2
  assert "scaffold" in item2["suggestion"]


def test_audit_json_cli_integration(tmp_path, capsys):
  """
  Scenario: Run via CLI entry point logic to verify argument parsing.
  """
  f = tmp_path / "simple.py"
  f.write_text("import torch\nx = torch.abs(y)")

  mock_mgr = MockAuditSemantics({"torch.abs": "torch"})

  with patch("ml_switcheroo.cli.handlers.audit.SemanticsManager", return_value=mock_mgr):
    # Run: ml_switcheroo audit path --roots torch --json
    args = ["audit", str(f), "--roots", "torch", "--json"]
    main(args)

  output = capsys.readouterr().out
  data = json.loads(output)
  assert len(data) == 1
  assert data[0]["api"] == "torch.abs"


def test_audit_json_valid_exit_code(tmp_path):
  """Verify correct exit codes exist even in JSON mode."""
  f = tmp_path / "valid.py"
  f.write_text("import torch\nx = torch.abs(y)")

  mock_mgr = MockAuditSemantics({"torch.abs": "torch"})

  with patch("ml_switcheroo.cli.handlers.audit.SemanticsManager", return_value=mock_mgr):
    # Should return 0 (success/clean)
    ret = handle_audit(f, ["torch"], json_mode=True)
    assert ret == 0


def test_audit_json_missing_exit_code(tmp_path):
  """Verify failure exit code if missing ops are found."""
  f = tmp_path / "bad.py"
  f.write_text("import torch\nx = torch.bad(y)")

  mock_mgr = MockAuditSemantics({})

  with patch("ml_switcheroo.cli.handlers.audit.SemanticsManager", return_value=mock_mgr):
    # Should return 1 (issues found)
    ret = handle_audit(f, ["torch"], json_mode=True)
    assert ret == 1
