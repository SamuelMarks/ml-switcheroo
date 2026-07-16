"""Tests for Data-Driven Self-Healing Patcher."""

from ml_switcheroo.testing.patcher import patch_json_spec
import json
import tempfile
from pathlib import Path


def test_patch_json_spec():
  """Auto-generated doc."""
  with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
    json.dump({"MyOp": {"variants": {"jax": {}}}}, f)
    path = Path(f.name)

  assert patch_json_spec(path, "MyOp", "jax", 0.01) is True

  with open(path, "r") as f:
    data = json.load(f)

  assert data["MyOp"]["test_rtol"] == 0.01
  assert data["MyOp"]["test_atol"] == 0.01

  assert patch_json_spec(path, "MissingOp", "jax", 0.01) is False
  path.unlink()


def test_patch_json_spec_error():
  """Auto-generated doc."""
  assert patch_json_spec(Path("/invalid/path/that/does/not/exist.json"), "Op", "jax", 0.1) is False
