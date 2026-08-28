"""Test suite for the Injector Fw module."""

import json
import pytest
from unittest.mock import patch
from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.tools.injector_fw import FrameworkInjector
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def target_json(tmp_path: Path) -> Path:
  """Provides a mock target JSON for testing."""
  defs_dir: Path = tmp_path / "definitions"
  defs_dir.mkdir()
  initial_data: Dict[str, Dict[str, str]] = {"OldOp": {"api": "torch.old"}}
  json_path: Path = defs_dir / "torch.json"
  json_path.write_text(json.dumps(initial_data), encoding="utf-8")
  return json_path


@pytest.fixture
def sample_variant() -> FrameworkVariant:
  """Provides a mock sample variant for testing."""
  return FrameworkVariant(api="torch.nn.functional.log_softmax", args={"dim": "dim"}, requires_plugin="custom_plugin")


def test_injector_updates_existing_json(target_json: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies the behavior of injector updates existing JSON."""
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    success: bool = injector.inject(dry_run=False)
  assert success is True
  content: Dict[str, Any] = json.loads(target_json.read_text())
  assert "OldOp" in content
  assert "LogSoftmax" in content
  entry: Dict[str, Any] = content["LogSoftmax"]
  assert entry["api"] == "torch.nn.functional.log_softmax"
  assert entry["args"] == {"dim": "dim"}
  assert entry["requires_plugin"] == "custom_plugin"


def test_injector_creates_new_file(tmp_path: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies the behavior of injector creates new file."""
  missing_path: Path = tmp_path / "definitions" / "new_fw.json"
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=missing_path):
    injector: FrameworkInjector = FrameworkInjector("new_fw", "NewOp", sample_variant)
    injector.inject(dry_run=False)
  assert missing_path.exists()
  content: Dict[str, Any] = json.loads(missing_path.read_text())
  assert "NewOp" in content
  assert content["NewOp"]["api"] == "torch.nn.functional.log_softmax"


def test_injector_dry_run(
  target_json: Path, sample_variant: FrameworkVariant, capsys: pytest.CaptureFixture[str]
) -> None:
  """Verifies the behavior of injector dry run."""
  target_json.stat().st_mtime
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    injector.inject(dry_run=True)
  captured: pytest.CaptureResult[str] = capsys.readouterr()
  assert "[Dry Run] Writing to torch.json" in captured.out
  assert "LogSoftmax" in captured.out
  content: Dict[str, Any] = json.loads(target_json.read_text())
  assert "LogSoftmax" not in content


def test_injector_idempotency(target_json: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies behavior when entry already exists and is identical."""
  # Add the sample_variant to target_json
  content: Dict[str, Any] = json.loads(target_json.read_text())
  content["LogSoftmax"] = sample_variant.model_dump(exclude_none=True)
  target_json.write_text(json.dumps(content))

  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    success: bool = injector.inject(dry_run=False)
    assert success is True


def test_injector_update_existing_different(target_json: Path, sample_variant: FrameworkVariant) -> None:
  """Hits line 65 where existing entry is updated."""
  # Add the LogSoftmax but with different data
  content: Dict[str, Any] = json.loads(target_json.read_text())
  content["LogSoftmax"] = {"api": "torch.wrong"}
  target_json.write_text(json.dumps(content))

  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    success: bool = injector.inject(dry_run=False)
    assert success is True
    new_content: Dict[str, Any] = json.loads(target_json.read_text())
    assert new_content["LogSoftmax"]["api"] == "torch.nn.functional.log_softmax"


def test_injector_write_error(target_json: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies error handling during file write."""
  original_open: Any = open

  def mock_open(*args: Any, **kwargs: Any) -> Any:
    """Mock open."""
    if len(args) > 1 and args[1] == "w":
      raise OSError("Permission denied")
    return original_open(*args, **kwargs)

  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    with patch("builtins.open", side_effect=mock_open):
      injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
      success: bool = injector.inject(dry_run=False)
      assert success is False


def test_injector_load_corrupt_json(target_json: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies behavior when loading corrupt JSON."""
  target_json.write_text("{invalid json}")
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    success: bool = injector.inject(dry_run=False)
    assert success is True
    # The file should be overwritten
    content: Dict[str, Any] = json.loads(target_json.read_text())
    assert "LogSoftmax" in content
    assert "OldOp" not in content
