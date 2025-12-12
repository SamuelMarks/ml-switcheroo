"""
Tests for Varargs Support (Updated).
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.discovery.inspector import ApiInspector


class MockVarargsInspector(ApiInspector):
  def inspect(self, fw_name):
    if fw_name == "source_fw":
      return {"source.add": {"name": "add", "params": ["x", "y"], "has_varargs": False}}
    if fw_name == "target_fw":
      return {"target.poly.add": {"name": "add", "params": ["args"], "has_varargs": True}}
    return {}


def test_scaffolder_skips_penalty_for_varargs(tmp_path):
  clean_semantics = SemanticsManager()
  clean_semantics.data = {}
  # Important: Clear origins so we don't fuzzy-match against real loaded specs "Add" vs "add"
  clean_semantics._key_origins = {}

  scaffolder = Scaffolder(semantics=clean_semantics, similarity_threshold=0.8, arity_penalty=0.5)
  scaffolder.inspector = MockVarargsInspector()

  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()

  scaffolder.scaffold(["source_fw", "target_fw"], root_dir=tmp_path)

  target_map = json.loads((snap_dir / "target_fw_vlatest_map.json").read_text())

  assert "add" in target_map["mappings"]
  assert target_map["mappings"]["add"]["api"] == "target.poly.add"
