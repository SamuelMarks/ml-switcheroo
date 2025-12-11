"""
Tests for Updater (Updated for Distributed Semantics).
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.updater import MappingsUpdater
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInspector:
  def inspect(self, _pkg):
    return {"pkg.nn.Layer": {"params": ["x"], "docstring_summary": "L"}}


class MockSemantics(SemanticsManager):
  def __init__(self):
    self.data = {}
    self._reverse_index = {}
    # Simulate empty so we find gaps

  def get_definition(self, api):
    return None


@pytest.fixture
def updater():
  upd = MappingsUpdater(MockSemantics())
  upd.inspector = MockInspector()
  return upd


def test_auto_merge_writes_to_disk(updater, tmp_path):
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"

  with patch("ml_switcheroo.discovery.updater.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.discovery.updater.resolve_snapshots_dir", return_value=snap_dir):
      updater.update_package("pkg", auto_merge=True)

  # Check Output
  neural_spec = sem_dir / "k_neural_net.json"
  overlay = snap_dir / "pkg_mappings.json"

  assert neural_spec.exists()
  assert overlay.exists()

  spec = json.loads(neural_spec.read_text())
  mapping = json.loads(overlay.read_text())

  # Layer should be present
  assert "Layer" in spec
  assert "variants" not in spec["Layer"]

  assert mapping["mappings"]["Layer"]["api"] == "pkg.nn.Layer"
