"""Test suite for the Patcher module."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from ml_switcheroo.testing.patcher import patch_json_spec


def test_patch_json_spec() -> None:
  """Patches JSON spec."""
  with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
    json.dump({"MyOp": {"variants": {"jax": {}}}}, f)
    path: Path = Path(f.name)
  assert patch_json_spec(path, "MyOp", "jax", 0.01) is True
  with open(path, "r") as f2:
    data: Dict[str, Any] = json.load(f2)
  assert data["MyOp"]["test_rtol"] == 0.01
  assert data["MyOp"]["test_atol"] == 0.01
  assert patch_json_spec(path, "MissingOp", "jax", 0.01) is False
  path.unlink()


def test_patch_json_spec_error() -> None:
  """Patches JSON spec correctly handling an error."""
  assert patch_json_spec(Path("/invalid/path/that/does/not/exist.json"), "Op", "jax", 0.1) is False
