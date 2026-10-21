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
        category: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if self._mode == InitMode.GHOST:
      if not self._snapshot_data:
        return []
      raw_list = getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr({"v": getattr({"v": self}, "__getitem__")("v")._snapshot_data}, "__getitem__")("v").get
                    },
                    "__getitem__",
                  )("v")(getattr({"v": "categories"}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
                },
                "__getitem__",
              )("v").get
            },
            "__getitem__",
          )("v")(
            getattr({"v": getattr({"v": category}, "__getitem__")("v").value}, "__getitem__")("v"),
            getattr({"v": []}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v")
      from ml_switcheroo.core.ghost import GhostInspector

      _ret_v: list[GhostRef] = getattr(
        {
          "v": [
            getattr({"v": getattr({"v": GhostInspector}, "__getitem__")("v").hydrate}, "__getitem__")("v")(
              getattr({"v": item}, "__getitem__")("v")
            )
            for item in raw_list
          ]
        },
        "__getitem__",
      )("v")
      return _ret_v
    else:
      return [GhostRef(name="LiveObj", api_path="mockfw.LiveObj", kind="class", params=[])]


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> typing.Generator[Path, None, None]:
  """Execute snapshot dir helper logic.

  Args:
      tmp_path: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  (tmp_path / "snapshots").mkdir()
  tgt_dir: Path = tmp_path / "snapshots"
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", tgt_dir):
    yield tgt_dir


@pytest.fixture
def valid_snapshot(snapshot_dir: Path) -> dict[str, JsonValue]:
  """Execute valid snapshot helper logic.

  Args:
      snapshot_dir: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  data = {
    "version": "1.0",
    "categories": {
      "loss": [{"name": "LiveObj", "api_path": "mockfw.LiveObj", "kind": "class", "params": [], "group": "class"}]
    },
  }
  file_path: Path = snapshot_dir / "mockfw_v1.0.json"
  file_path.write_text(json.dumps(data))
  (snapshot_dir / "mockfw_v0.9.json").write_text(json.dumps(data))
  _ret_v: dict[str, JsonValue] = getattr({"v": data}, "__getitem__")("v")
  return _ret_v


def test_load_snapshot_helper_finds_latest(valid_snapshot: dict[str, JsonValue]) -> None:
  """Verify load snapshot helper finds latest behavior.

  Args:
      valid_snapshot: Test argument or fixture.
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
      valid_snapshot: Test argument or fixture.
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
      snapshot_dir: Test argument or fixture.
  """
  with patch.dict(sys.modules):
    if "mockfw" in sys.modules:
      del sys.modules["mockfw"]
    adapter = MockAdapter()
    assert adapter._mode == InitMode.GHOST
    assert adapter._snapshot_data == {}
    results: list[GhostRef] = adapter.collect_api(SemanticTier.LOSS)
    assert results == []
