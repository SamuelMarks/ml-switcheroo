"""Test suite for the Ghost Hydrating module."""

import json
import pytest
import sys
from unittest.mock import MagicMock, patch
from ml_switcheroo.frameworks.base import SemanticTier, GhostRef, InitMode, load_snapshot_for_adapter


class MockAdapter:
  """Mock Adapter class for testing purposes."""

  def __init__(self):
    """Initializes the MockAdapter instance."""
    self._mode = InitMode.LIVE
    self._snapshot_data = {}
    if "mockfw" not in sys.modules:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("mockfw")

  def collect_api(self, category: SemanticTier) -> list[GhostRef]:
    """Mock implementation of collect API."""
    if self._mode == InitMode.GHOST:
      if not self._snapshot_data:
        return []
      raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
      from ml_switcheroo.core.ghost import GhostInspector

      return [GhostInspector.hydrate(item) for item in raw_list]
    else:
      return [GhostRef(name="LiveObj", api_path="mockfw.LiveObj", kind="class", params=[])]


@pytest.fixture
def snapshot_dir(tmp_path):
  """Provides a mock snapshot directory for testing."""
  (tmp_path / "snapshots").mkdir()
  tgt_dir = tmp_path / "snapshots"
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", tgt_dir):
    yield tgt_dir


@pytest.fixture
def valid_snapshot(snapshot_dir):
  """Provides a mock valid snapshot for testing."""
  data = {
    "version": "1.0",
    "categories": {"loss": [{"name": "LiveObj", "api_path": "mockfw.LiveObj", "kind": "class", "params": []}]},
  }
  file_path = snapshot_dir / "mockfw_v1.0.json"
  file_path.write_text(json.dumps(data))
  (snapshot_dir / "mockfw_v0.9.json").write_text(json.dumps(data))
  return data


def test_load_snapshot_helper_finds_latest(valid_snapshot):
  """Loads snapshot helper finds latest."""
  data = load_snapshot_for_adapter("mockfw")
  assert data is not None
  assert data["version"] == "1.0"


def test_hybrid_mode_live():
  """Verifies the behavior of hybrid mode live."""
  with patch.dict(sys.modules, {"mockfw": MagicMock()}):
    adapter = MockAdapter()
    assert adapter._mode == InitMode.LIVE
    results = adapter.collect_api(SemanticTier.LOSS)
    assert len(results) == 1
    assert results[0].name == "LiveObj"


def test_hybrid_mode_ghost(valid_snapshot):
  """Verifies the behavior of hybrid mode ghost."""
  with patch.dict(sys.modules):
    if "mockfw" in sys.modules:
      del sys.modules["mockfw"]
    adapter = MockAdapter()
    assert adapter._mode == InitMode.GHOST
    assert adapter._snapshot_data["version"] == "1.0"
    results = adapter.collect_api(SemanticTier.LOSS)
    assert len(results) == 1
    ref = results[0]
    assert type(ref).__name__ == "GhostRef"
    assert ref.name == "LiveObj"
    assert ref.api_path == "mockfw.LiveObj"


def test_ghost_mode_no_snapshot(snapshot_dir):
  """Verifies the behavior of ghost mode no snapshot."""
  with patch.dict(sys.modules):
    if "mockfw" in sys.modules:
      del sys.modules["mockfw"]
    adapter = MockAdapter()
    assert adapter._mode == InitMode.GHOST
    assert adapter._snapshot_data == {}
    results = adapter.collect_api(SemanticTier.LOSS)
    assert results == []
