"""Test suite for AdapterRegistry, SnapshotBackedAdapter, and framework registration helpers."""

import json
import logging
import pathlib
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import JsonValue

from ml_switcheroo.exceptions import ConfigurationError
from ml_switcheroo.frameworks.base import (
  AdapterRegistry,
  InitMode,
  SemanticTier,
  SnapshotBackedAdapter,
  available_frameworks,
  get_adapter,
  get_adapter_class,
  get_default_adapter_registry,
  load_snapshot_for_adapter,
  register_framework,
)
from ml_switcheroo.frameworks.ir import IrAdapter


def test_available_frameworks() -> None:
  """Verify ``available_frameworks`` lists newly registered adapter keys."""
  key = "test_framework_avail"

  @register_framework(key)
  class TestAdapterAvail:
    """Stub framework adapter for availability check."""

  available = available_frameworks()
  assert key in available
  assert isinstance(available, list)


def test_registry_mechanics_and_caching() -> None:
  """Verify ``AdapterRegistry`` registration, class lookup, instance caching, and ``ConfigurationError`` guard."""
  key = "test_framework_base"

  @register_framework(key)
  class TestAdapter(IrAdapter):
    """Stub framework adapter tracking initialization."""

    def __init__(self) -> None:
      """Record initialization on the adapter instance."""
      super().__init__()
      self.initialized = True

  reg = get_default_adapter_registry()
  assert key in reg.adapters
  assert get_adapter_class(key) is TestAdapter
  first = get_adapter(key)
  second = get_adapter(key)
  assert isinstance(first, TestAdapter)
  assert first.initialized is True
  assert first is second

  reg.clear_cache()
  third = get_adapter(key)
  assert isinstance(third, TestAdapter)
  assert third is not first

  isolated = AdapterRegistry()
  with pytest.raises(ConfigurationError, match="must be a class"):
    isolated.register("invalid")("not_a_class")


def test_get_adapter_missing() -> None:
  """Verify ``get_adapter`` returns ``None`` for unregistered framework keys."""
  assert get_adapter("non_existent_framework") is None
  assert get_adapter_class("non_existent_framework") is None


def test_snapshot_backed_adapter_modes(caplog: pytest.LogCaptureFixture) -> None:
  """Verify ``SnapshotBackedAdapter`` live/ghost transitions and ``FrameworkSnapshot`` hydration.

  Args:
      caplog: Pytest log capture fixture.
  """
  from ml_switcheroo.frameworks.base import FrameworkSnapshot, GhostRef

  live = SnapshotBackedAdapter("demo", is_live_available=True)
  assert live.mode == InitMode.LIVE
  assert live.snapshot == FrameworkSnapshot()
  assert live.snapshot.is_empty() is True
  assert live.collect_ghost(SemanticTier.NEURAL) == []

  with caplog.at_level(logging.DEBUG):
    ghost_empty = SnapshotBackedAdapter(
      "demo",
      is_live_available=False,
      loader=lambda _: None,
      missing_message="demo snapshot missing",
    )
    assert ghost_empty.mode == InitMode.GHOST
    assert ghost_empty.snapshot == FrameworkSnapshot()
    assert "demo snapshot missing" in caplog.text

    ghost_model = SnapshotBackedAdapter(
      "demo",
      is_live_available=False,
      loader=lambda _: FrameworkSnapshot(
        version="2.0",
        categories={SemanticTier.NEURAL.value: [GhostRef(name="Linear", api_path="demo.nn.Linear", kind="class")]},
      ),
      missing_message="should not log",
    )
    assert ghost_model.snapshot.is_empty() is False
    assert ghost_model.snapshot.version == "2.0"
    assert len(ghost_model.collect_ghost(SemanticTier.NEURAL)) == 1

  assert SnapshotBackedAdapter.collect_ghost_from(None, SemanticTier.NEURAL) == []
  assert SnapshotBackedAdapter.collect_ghost_from({"categories": "invalid"}, SemanticTier.NEURAL) == []
  assert SnapshotBackedAdapter.collect_ghost_from({"categories": {"neural": "invalid"}}, SemanticTier.NEURAL) == []

  valid_snap: dict[str, JsonValue] = {
    "version": "1.2",
    "categories": {
      SemanticTier.NEURAL.value: [
        {"name": "Linear", "api_path": "demo.nn.Linear", "kind": "class", "args": []},
        "ignored_non_dict_entry",
      ]
    },
  }
  refs = SnapshotBackedAdapter.collect_ghost_from(valid_snap, SemanticTier.NEURAL)
  assert len(refs) == 1
  assert refs[0].name == "Linear"
  mixed_snap = FrameworkSnapshot(categories={SemanticTier.NEURAL.value: [refs[0]]})
  assert len(mixed_snap.get_category(SemanticTier.NEURAL)) == 1
  direct_refs = SnapshotBackedAdapter.collect_ghost_from(ghost_model.snapshot, SemanticTier.NEURAL)
  assert len(direct_refs) == 1


@pytest.fixture
def mock_snapshot_dir(tmp_path: pathlib.Path) -> pathlib.Path:
  """Create an isolated snapshot directory under ``tmp_path``.

  Args:
      tmp_path: Pytest temporary directory path fixture.

  Returns:
      Created snapshot directory path.
  """
  d = tmp_path / "snapshots"
  d.mkdir()
  return d


def test_load_snapshot_sorts_versions(mock_snapshot_dir: pathlib.Path) -> None:
  """Verify ``load_snapshot_for_adapter`` selects the highest lexical version file.

  Args:
      mock_snapshot_dir: Temporary directory populated with versioned snapshot files.
  """
  (mock_snapshot_dir / "testfw_v1.0.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
  (mock_snapshot_dir / "testfw_v2.0.json").write_text(json.dumps({"version": "2.0"}), encoding="utf-8")
  (mock_snapshot_dir / "testfw_v1.5.json").write_text(json.dumps({"version": "1.5"}), encoding="utf-8")
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir):
    result = load_snapshot_for_adapter("testfw")
  assert result["version"] == "2.0"


def test_load_snapshot_missing_dir() -> None:
  """Verify ``load_snapshot_for_adapter`` returns ``{}`` when ``SNAPSHOT_DIR`` does not exist."""
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", Path("/non/existent")):
    result = load_snapshot_for_adapter("any")
    assert result == {}


def test_load_snapshot_no_match(mock_snapshot_dir: pathlib.Path) -> None:
  """Verify ``load_snapshot_for_adapter`` returns ``{}`` when no matching snapshot file exists.

  Args:
      mock_snapshot_dir: Empty temporary snapshot directory.
  """
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir):
    result = load_snapshot_for_adapter("other_fw")
    assert result == {}


def test_load_snapshot_corrupt_file(mock_snapshot_dir: pathlib.Path, caplog: pytest.LogCaptureFixture) -> None:
  """Verify ``load_snapshot_for_adapter`` logs an error and returns ``{}`` on corrupt JSON.

  Args:
      mock_snapshot_dir: Temporary snapshot directory containing malformed JSON.
      caplog: Pytest log capture fixture.
  """
  (mock_snapshot_dir / "corrupt_v1.json").write_text("{bad json", encoding="utf-8")
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir), caplog.at_level(logging.ERROR):
    result = load_snapshot_for_adapter("corrupt")
    assert result == {}
    assert "Failed to load snapshot" in caplog.text
