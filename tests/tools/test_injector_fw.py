"""
Tests for the Framework Definition Injector (JSON Mode).

Verifies that:
1. It loads existing JSON definitions.
2. It inserts or updates operation mappings.
3. It handles missing files gracefully (creates them).
4. Dry run mode does not write to disk.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.tools.injector_fw import FrameworkInjector


@pytest.fixture
def target_json(tmp_path):
  """Creates a temporary definition directory and file."""
  defs_dir = tmp_path / "definitions"
  defs_dir.mkdir()

  # Pre-populate with one op
  initial_data = {"OldOp": {"api": "torch.old"}}
  json_path = defs_dir / "torch.json"
  json_path.write_text(json.dumps(initial_data), encoding="utf-8")

  return json_path


@pytest.fixture
def sample_variant():
  return FrameworkVariant(
    api="torch.nn.functional.log_softmax",
    args={"dim": "dim"},
    requires_plugin="custom_plugin",
  )


def test_injector_updates_existing_json(target_json, sample_variant):
  """
  Scenario: File exists. Inject a new Op.
  """
  # Patch the path discovery to point to our temp file
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    success = injector.inject(dry_run=False)

  assert success is True

  # Verify Content
  content = json.loads(target_json.read_text())
  assert "OldOp" in content
  assert "LogSoftmax" in content

  entry = content["LogSoftmax"]
  assert entry["api"] == "torch.nn.functional.log_softmax"
  assert entry["args"] == {"dim": "dim"}
  assert entry["requires_plugin"] == "custom_plugin"


def test_injector_creates_new_file(tmp_path, sample_variant):
  """
  Scenario: Definitions file does not exist.
  """
  missing_path = tmp_path / "definitions" / "new_fw.json"

  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=missing_path):
    injector = FrameworkInjector("new_fw", "NewOp", sample_variant)
    injector.inject(dry_run=False)

  assert missing_path.exists()
  content = json.loads(missing_path.read_text())
  assert "NewOp" in content
  assert content["NewOp"]["api"] == "torch.nn.functional.log_softmax"


def test_injector_dry_run(target_json, sample_variant, capsys):
  """
  Scenario: Dry Run enabled. File should NOT change.
  """
  original_mtime = target_json.stat().st_mtime

  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    injector.inject(dry_run=True)

  # check stdout
  captured = capsys.readouterr()
  assert "[Dry Run] Writing to torch.json" in captured.out
  assert "LogSoftmax" in captured.out

  # check file untouched
  content = json.loads(target_json.read_text())
  assert "LogSoftmax" not in content
