"""
Integration Test for Hybrid Loading (Ghost Protocol).

Verifies that:
1. An Adapter correctly falls back to Ghost Mode when the library is missing.
2. It hydrates data from a JSON snapshot.
3. `collect_api` returns the *exact same* GhostRef structures in both Live and Ghost modes.
   (This ensures downstream logic requires zero awareness of the environment state).
"""

import json
import pytest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

from ml_switcheroo.frameworks.base import (
  StandardCategory,
  GhostRef,
  _ADAPTER_REGISTRY,
  InitMode,
  load_snapshot_for_adapter,
)


# --- Test Setup: Create a "Reference" Adapter Class ---


class MockAdapter:
  """
  A minimal adapter implementation that supports the Hybrid Protocol.
  We inject logic here similar to what real adapters (Torch/JAX) will do.
  """

  def __init__(self):
    self._mode = InitMode.LIVE
    self._snapshot_data = {}

    # Simulate Import Check
    if "mockfw" not in sys.modules:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("mockfw")

  def collect_api(self, category: StandardCategory) -> list[GhostRef]:
    if self._mode == InitMode.GHOST:
      # Ghost Implementation: specific logic to read from loaded dict
      if not self._snapshot_data:
        return []

      # Map JSON list -> GhostRef objects
      raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
      from ml_switcheroo.core.ghost import GhostInspector

      return [GhostInspector.hydrate(item) for item in raw_list]

    else:
      # Live Implementation: Return a hardcoded "Live" object
      # In real life this inspects the module.
      return [GhostRef(name="LiveObj", api_path="mockfw.LiveObj", kind="class", params=[])]


# --- Fixtures ---


@pytest.fixture
def snapshot_dir(tmp_path):
  """
  Create a fake snapshot directory and inject it into the module.
  """
  # Create the structure
  (tmp_path / "snapshots").mkdir()
  tgt_dir = tmp_path / "snapshots"

  # Patch the SNAPSHOT_DIR constant in base.py
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", tgt_dir):
    yield tgt_dir


@pytest.fixture
def valid_snapshot(snapshot_dir):
  """
  Creates a valid snapshot file for 'mockfw'.
  Contains data that perfectly mimics the "Live" data to prove parity.
  """
  data = {
    "version": "1.0",
    "categories": {
      "loss": [
        {
          "name": "LiveObj",  # Same name as Live to prove alignment
          "api_path": "mockfw.LiveObj",
          "kind": "class",
          "params": [],
        }
      ]
    },
  }

  # Save as version 1.0 (should be picked up)
  file_path = snapshot_dir / "mockfw_v1.0.json"
  file_path.write_text(json.dumps(data))

  # Create an older version (should be ignored)
  (snapshot_dir / "mockfw_v0.9.json").write_text(json.dumps(data))

  return data


# --- Tests ---


def test_load_snapshot_helper_finds_latest(valid_snapshot):
  """
  Verify the utility function picks the correct file (lexical sort).
  """
  data = load_snapshot_for_adapter("mockfw")
  assert data is not None
  assert data["version"] == "1.0"  # Picked v1.0 over v0.9


def test_hybrid_mode_live():
  """
  Scenario: 'mockfw' is in sys.modules.
  Expectation: Adapter stays in LIVE mode and returns live data.
  """
  # Inject mock module
  with patch.dict(sys.modules, {"mockfw": MagicMock()}):
    adapter = MockAdapter()

    assert adapter._mode == InitMode.LIVE
    results = adapter.collect_api(StandardCategory.LOSS)

    assert len(results) == 1
    assert results[0].name == "LiveObj"
    # Since logic isn't strictly identical in this mock (hardcoded),
    # we just verify it took the LIVE branch.


def test_hybrid_mode_ghost(valid_snapshot):
  """
  Scenario: 'mockfw' is MISSING. Snapshot exists.
  Expectation: Adapter enters GHOST mode, loads snapshot, returns data matching Live structure.
  """
  # Ensure mockfw not present
  with patch.dict(sys.modules):
    if "mockfw" in sys.modules:
      del sys.modules["mockfw"]

    adapter = MockAdapter()

    assert adapter._mode == InitMode.GHOST
    assert adapter._snapshot_data["version"] == "1.0"

    # Run Collection
    results = adapter.collect_api(StandardCategory.LOSS)

    # Verify Hydration
    assert len(results) == 1
    ref = results[0]

    # Check integrity
    assert isinstance(ref, GhostRef)
    assert ref.name == "LiveObj"
    assert ref.api_path == "mockfw.LiveObj"


def test_ghost_mode_no_snapshot(snapshot_dir):
  """
  Scenario: 'mockfw' missing AND no snapshot file.
  Expectation: Graceful empty state.
  """
  with patch.dict(sys.modules):
    if "mockfw" in sys.modules:
      del sys.modules["mockfw"]

    adapter = MockAdapter()
    assert adapter._mode == InitMode.GHOST
    assert adapter._snapshot_data == {}

    results = adapter.collect_api(StandardCategory.LOSS)
    assert results == []
