"""Test suite for the Injector Fw module."""

import json
import pytest
from unittest.mock import patch
from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.tools.injector_fw import FrameworkInjector


@pytest.fixture
def target_json(tmp_path):
  """Provides a mock target JSON for testing."""
  defs_dir = tmp_path / "definitions"
  defs_dir.mkdir()
  initial_data = {"OldOp": {"api": "torch.old"}}
  json_path = defs_dir / "torch.json"
  json_path.write_text(json.dumps(initial_data), encoding="utf-8")
  return json_path


@pytest.fixture
def sample_variant():
  """Provides a mock sample variant for testing."""
  return FrameworkVariant(api="torch.nn.functional.log_softmax", args={"dim": "dim"}, requires_plugin="custom_plugin")


def test_injector_updates_existing_json(target_json, sample_variant):
  """Verifies the behavior of injector updates existing JSON."""
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    success = injector.inject(dry_run=False)
  assert success is True
  content = json.loads(target_json.read_text())
  assert "OldOp" in content
  assert "LogSoftmax" in content
  entry = content["LogSoftmax"]
  assert entry["api"] == "torch.nn.functional.log_softmax"
  assert entry["args"] == {"dim": "dim"}
  assert entry["requires_plugin"] == "custom_plugin"


def test_injector_creates_new_file(tmp_path, sample_variant):
  """Verifies the behavior of injector creates new file."""
  missing_path = tmp_path / "definitions" / "new_fw.json"
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=missing_path):
    injector = FrameworkInjector("new_fw", "NewOp", sample_variant)
    injector.inject(dry_run=False)
  assert missing_path.exists()
  content = json.loads(missing_path.read_text())
  assert "NewOp" in content
  assert content["NewOp"]["api"] == "torch.nn.functional.log_softmax"


def test_injector_dry_run(target_json, sample_variant, capsys):
  """Verifies the behavior of injector dry run."""
  target_json.stat().st_mtime
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    injector.inject(dry_run=True)
  captured = capsys.readouterr()
  assert "[Dry Run] Writing to torch.json" in captured.out
  assert "LogSoftmax" in captured.out
  content = json.loads(target_json.read_text())
  assert "LogSoftmax" not in content
