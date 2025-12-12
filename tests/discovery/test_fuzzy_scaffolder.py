"""
Tests for Fuzzy Matching (Updated for Distributed Semantics).
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.discovery.inspector import ApiInspector
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInspector(ApiInspector):
  def inspect(self, fw_name: str) -> dict:
    if fw_name == "source_fw":
      return {
        "source.absolute": {"name": "absolute", "params": ["x"]},
        "source.unary_op": {"name": "unary_op", "params": ["x"]},
      }
    if fw_name == "target_fw":
      return {
        "target.abs": {"name": "abs", "params": ["a"]},
        "target.add": {"name": "add", "params": ["a", "b"]},
        "target.wrong_arity_op": {"name": "unary_op", "params": ["a", "b", "c"]},
      }
    return {}


@pytest.fixture
def clean_semantics():
  mgr = SemanticsManager()
  mgr._reverse_index = {}
  mgr.data = {}
  mgr._key_origins = {}
  return mgr


def test_fuzzy_match_success(tmp_path, clean_semantics):
  scaffolder = Scaffolder(semantics=clean_semantics, similarity_threshold=0.6)
  scaffolder.inspector = MockInspector()

  # Configure Paths
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()

  # Explicitly path resolving isn't needed if we pass root_dir=tmp_path
  # because Scaffolder appends /semantics and /snapshots automatically.
  # Using tmp_path as root ensures files land in tmp_path/semantics and tmp_path/snapshots.

  # Scaffolder.scaffold writes to importlib.metadata.version or 'latest'
  # We need to expect 'latest' since frameworks aren't installed.

  scaffolder.scaffold(["source_fw", "target_fw"], root_dir=tmp_path)

  # Check Mapping File for Target
  tgt_file = snap_dir / "target_fw_vlatest_map.json"
  assert tgt_file.exists()

  data = json.loads(tgt_file.read_text())
  mappings = data["mappings"]

  assert "absolute" in mappings
  assert mappings["absolute"]["api"] == "target.abs"


def test_signature_analysis_rejection(tmp_path, clean_semantics):
  scaffolder = Scaffolder(semantics=clean_semantics, similarity_threshold=0.8, arity_penalty=0.5)
  scaffolder.inspector = MockInspector()

  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()

  scaffolder.scaffold(["source_fw", "target_fw"], root_dir=tmp_path)

  tgt_file = snap_dir / "target_fw_vlatest_map.json"
  if tgt_file.exists():
    data = json.loads(tgt_file.read_text())
    mappings = data["mappings"]
    assert "unary_op" not in mappings


def test_exact_match_priority(tmp_path, clean_semantics):
  scaffolder = Scaffolder(semantics=clean_semantics)
  scaffolder.inspector = MockInspector()

  def inspect_side(fw_name):
    if fw_name == "source_fw":
      return {"source.add": {"name": "add", "params": ["x", "y"]}}
    else:
      return {"target.add": {"name": "add", "params": ["a", "b"]}}

  scaffolder.inspector.inspect = MagicMock(side_effect=inspect_side)

  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()

  scaffolder.scaffold(["source_fw", "target_fw"], root_dir=tmp_path)

  data = json.loads((snap_dir / "target_fw_vlatest_map.json").read_text())

  assert "add" in data["mappings"]
  assert data["mappings"]["add"]["api"] == "target.add"
