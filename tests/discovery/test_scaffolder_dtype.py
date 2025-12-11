"""
Tests for Scaffolder Dtype support (Split Write).
"""

import json
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


class MockAttributeInspector:
  def inspect(self, fw):
    if fw == "torch":
      return {"torch.float32": {"name": "float32", "type": "attribute", "params": []}}
    return {}


def test_scaffolder_propagates_type_field(tmp_path):
  clean_semantics = SemanticsManager()
  clean_semantics.data = {}
  # Fix: Initialize missing attributes
  clean_semantics._key_origins = {}

  scaffolder = Scaffolder(semantics=clean_semantics)
  scaffolder.inspector = MockAttributeInspector()

  # Use subdir structure to match logical expectation of sibling snapshots
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()

  with patch("ml_switcheroo.discovery.scaffolder.available_frameworks", return_value=["torch"]):
    # Patch adapter to avoid real torch lookup
    with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=MagicMock()):
      scaffolder.scaffold(["torch"], sem_dir)

  # Check Spec
  spec = json.loads((sem_dir / "k_array_api.json").read_text())
  assert spec["float32"]["type"] == "attribute"

  # Check Mapping
  snap = json.loads((snap_dir / "torch_mappings.json").read_text())
  assert snap["mappings"]["float32"]["api"] == "torch.float32"
