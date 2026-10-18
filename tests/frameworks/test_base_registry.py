"""Test suite for the Base Registry module."""

import json
import logging
import typing
from pathlib import Path
from unittest.mock import patch

import pytest

from ml_switcheroo.frameworks.base import (
  _ADAPTER_REGISTRY,
  available_frameworks,
  get_adapter,
  load_snapshot_for_adapter,
  register_framework,
)


def test_available_frameworks() -> None:
  """Verifies the behavior of available_frameworks."""
  key: typing.Any = "test_framework_avail"

  @register_framework(key)
  class TestAdapterAvail:
    """Test helper class TestAdapterAvail."""

    pass

  available: typing.Any = available_frameworks()
  assert key in available
  assert isinstance(available, list)


def test_registry_mechanics() -> None:
  """Verifies the behavior of registry mechanics."""
  key: typing.Any = "test_framework_base"

  @register_framework(key)
  class TestAdapter:
    """Test helper class TestAdapter."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
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
def mock_snapshot_dir(tmp_path: typing.Any) -> typing.Any:
  """Execute mock snapshot dir helper logic.

  Args:
      tmp_path: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  d: typing.Any = tmp_path / "snapshots"
  d.mkdir()
  return d


def test_load_snapshot_sorts_versions(mock_snapshot_dir: typing.Any) -> None:
  """Loads snapshot sorts versions.

  Args:
      mock_snapshot_dir: Test argument or fixture.
  """
  (mock_snapshot_dir / "testfw_v1.0.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
  (mock_snapshot_dir / "testfw_v2.0.json").write_text(json.dumps({"version": "2.0"}), encoding="utf-8")
  (mock_snapshot_dir / "testfw_v1.5.json").write_text(json.dumps({"version": "1.5"}), encoding="utf-8")
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir):
    result: typing.Any = load_snapshot_for_adapter("testfw")
  assert result["version"] == "2.0"


def test_load_snapshot_missing_dir() -> None:
  """Loads snapshot missing directory."""
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", Path("/non/existent")):
    result: typing.Any = load_snapshot_for_adapter("any")
    assert result == {}


def test_load_snapshot_no_match(mock_snapshot_dir: typing.Any) -> None:
  """Loads snapshot no match.

  Args:
      mock_snapshot_dir: Test argument or fixture.
  """
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir):
    result: typing.Any = load_snapshot_for_adapter("other_fw")
    assert result == {}


def test_load_snapshot_corrupt_file(mock_snapshot_dir: typing.Any, caplog: typing.Any) -> None:
  """Loads snapshot corrupt file.

  Args:
      mock_snapshot_dir: Test argument or fixture.
      caplog: Test argument or fixture.
  """
  (mock_snapshot_dir / "corrupt_v1.json").write_text("{bad json", encoding="utf-8")
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir):
    with caplog.at_level(logging.ERROR):
      result: typing.Any = load_snapshot_for_adapter("corrupt")
      assert result == {}
      assert "Failed to load snapshot" in caplog.text
