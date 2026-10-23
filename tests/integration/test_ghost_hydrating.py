"""Test suite for the Ghost Hydrating module."""

import json
import sys
import typing
from pydantic import JsonValue
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_switcheroo.frameworks.base import InitMode, load_snapshot_for_adapter


class MockAdapter:
  """Test helper class MockAdapter."""

  def __init__(self) -> None:
    """Initializes the MockAdapter instance."""
    self._mode = InitMode.LIVE
    self._snapshot_data = {}
    if "mockfw" not in sys.modules:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("mockfw")

  def collect_api(self, category: SemanticTier) -> list[GhostRef]:
    """Mock implementation of collect API.

    Args:
        category: The category (SemanticTier) for collect api.

    Returns:
        Resulting `list[GhostRef]` from collect api.
    """
    if self._mode == InitMode.GHOST:
      if not self._snapshot_data:
        return []
      cats = self._snapshot_data.get("categories", {})
      raw_list = cats.get(category.value, []) if isinstance(cats, dict) else []
      if not isinstance(raw_list, list):
        return []
      from ml_switcheroo.core.ghost import GhostInspector

      return [GhostInspector.hydrate(item) for item in raw_list if isinstance(item, dict)]
    else:
      return [GhostRef(name="LiveObj", api_path="mockfw.LiveObj", kind="class", params=[])]


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> typing.Generator[Path, None, None]:
  """Execute snapshot dir helper logic.

  Args:
      tmp_path: Pytest temporary directory path fixture.

  Returns:
      Configured `snapshot_dir` fixture value (typing.Generator[Path, None, None]).
  """
  (tmp_path / "snapshots").mkdir()
  tgt_dir: Path = tmp_path / "snapshots"
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", tgt_dir):
    yield tgt_dir


@pytest.fixture
def valid_snapshot(snapshot_dir: Path) -> dict[str, JsonValue]:
  """Execute valid snapshot helper logic.

  Args:
      snapshot_dir: The snapshot dir (Path) for valid snapshot.

  Returns:
      Configured `valid_snapshot` fixture value (dict[str, JsonValue]).
  """
  data: dict[str, JsonValue] = {
    "version": "1.0",
    "categories": {
      "loss": [{"name": "LiveObj", "api_path": "mockfw.LiveObj", "kind": "class", "params": [], "group": "class"}]
    },
  }
  file_path: Path = snapshot_dir / "mockfw_v1.0.json"
  file_path.write_text(json.dumps(data))
  (snapshot_dir / "mockfw_v0.9.json").write_text(json.dumps(data))
  return data


def test_load_snapshot_helper_finds_latest(valid_snapshot: dict[str, JsonValue]) -> None:
  """Verify load snapshot helper finds latest behavior.

  Args:
      valid_snapshot: The valid snapshot (dict[str, JsonValue]) for test load snapshot helper finds latest.
  """
  data = load_snapshot_for_adapter("mockfw")
  assert data is not None
  assert data["version"] == "1.0"


def test_hybrid_mode_live() -> None:
  """Verifies the behavior of hybrid mode live."""
  with patch.dict(sys.modules, {"mockfw": MagicMock()}):
    adapter = MockAdapter()
    assert adapter._mode == InitMode.LIVE
    results: list[GhostRef] = adapter.collect_api(SemanticTier.LOSS)
    assert len(results) == 1
    assert results[0].name == "LiveObj"


def test_hybrid_mode_ghost(valid_snapshot: dict[str, JsonValue]) -> None:
  """Verifies the behavior of hybrid mode ghost.

  Args:
      valid_snapshot: The valid snapshot (dict[str, JsonValue]) for test hybrid mode ghost.
  """
  with patch.dict(sys.modules):
    if "mockfw" in sys.modules:
      del sys.modules["mockfw"]
    adapter = MockAdapter()
    assert adapter._mode == InitMode.GHOST
    assert adapter._snapshot_data["version"] == "1.0"
    results: list[GhostRef] = adapter.collect_api(SemanticTier.LOSS)
    assert len(results) == 1
    ref: GhostRef = results[0]
    assert type(ref).__name__ == "GhostRef"
    assert ref.name == "LiveObj"
    assert ref.api_path == "mockfw.LiveObj"


def test_ghost_mode_no_snapshot(snapshot_dir: Path) -> None:
  """Verifies the behavior of ghost mode no snapshot.

  Args:
      snapshot_dir: The snapshot dir (Path) for test ghost mode no snapshot.
  """
  with patch.dict(sys.modules):
    if "mockfw" in sys.modules:
      del sys.modules["mockfw"]
    adapter = MockAdapter()
    assert adapter._mode == InitMode.GHOST
    assert adapter._snapshot_data == {}
    results: list[GhostRef] = adapter.collect_api(SemanticTier.LOSS)
    assert results == []
