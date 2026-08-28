"""Test suite for the Base Registry module."""

import pytest
import json
import logging
import typing
from pathlib import Path
from unittest.mock import patch
from ml_switcheroo.frameworks.base import (
  register_framework,
  get_adapter,
  load_snapshot_for_adapter,
  _ADAPTER_REGISTRY,
  available_frameworks,
)


def test_available_frameworks() -> None:
  """Verifies the behavior of available_frameworks."""
  key = "test_framework_avail"

  @register_framework(key)
  class TestAdapterAvail:
    """Test Adapter."""

    pass

  available: list[str] = available_frameworks()
  assert key in available
  assert isinstance(available, list)


def test_registry_mechanics() -> None:
  """Verifies the behavior of registry mechanics."""
  key = "test_framework_base"

  @register_framework(key)
  class TestAdapter:
    """Test suite for the Adapter component."""

    def __init__(self) -> None:
      """Initializes the TestAdapter instance."""
      self.initialized = True

  assert key in _ADAPTER_REGISTRY
  assert _ADAPTER_REGISTRY[key] == TestAdapter
  instance: typing.Any = get_adapter(key)
  assert isinstance(instance, TestAdapter)
  assert instance.initialized is True


def test_get_adapter_missing() -> None:
  """Gets adapter missing."""
  assert get_adapter("non_existent_framework") is None


@pytest.fixture
def mock_snapshot_dir(tmp_path: Path) -> Path:
  """Provides a mock snapshot directory for testing."""
  d: Path = tmp_path / "snapshots"
  d.mkdir()
  return d


def test_load_snapshot_sorts_versions(mock_snapshot_dir: Path) -> None:
  """Loads snapshot sorts versions."""
  (mock_snapshot_dir / "testfw_v1.0.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
  (mock_snapshot_dir / "testfw_v2.0.json").write_text(json.dumps({"version": "2.0"}), encoding="utf-8")
  (mock_snapshot_dir / "testfw_v1.5.json").write_text(json.dumps({"version": "1.5"}), encoding="utf-8")
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir):
    result: dict[str, typing.Any] = load_snapshot_for_adapter("testfw")
  assert result["version"] == "2.0"


def test_load_snapshot_missing_dir() -> None:
  """Loads snapshot missing directory."""
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", Path("/non/existent")):
    result: dict[str, typing.Any] = load_snapshot_for_adapter("any")
    assert result == {}


def test_load_snapshot_no_match(mock_snapshot_dir: Path) -> None:
  """Loads snapshot no match."""
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir):
    result: dict[str, typing.Any] = load_snapshot_for_adapter("other_fw")
    assert result == {}


def test_load_snapshot_corrupt_file(mock_snapshot_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
  """Loads snapshot corrupt file."""
  (mock_snapshot_dir / "corrupt_v1.json").write_text("{bad json", encoding="utf-8")
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir):
    with caplog.at_level(logging.ERROR):
      result: dict[str, typing.Any] = load_snapshot_for_adapter("corrupt")
      assert result == {}
      assert "Failed to load snapshot" in caplog.text
