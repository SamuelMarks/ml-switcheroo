"""
Tests for Framework Registry and Base Protocol Mechanics.

Verifies:
1. Registration Decorator adds classes to `_ADAPTER_REGISTRY`.
2. `get_adapter` instantiates the class.
3. `load_snapshot_for_adapter` handles version sorting and missing files correctly.
"""

import pytest
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_switcheroo.frameworks.base import (
  register_framework,
  get_adapter,
  load_snapshot_for_adapter,
  _ADAPTER_REGISTRY,
)


# --- Registry Tests ---


def test_registry_mechanics():
  """Verify @register_framework decorator adds logic."""
  # Use a unique key for isolation
  key = "test_framework_base"

  @register_framework(key)
  class TestAdapter:
    def __init__(self):
      self.initialized = True

  assert key in _ADAPTER_REGISTRY
  assert _ADAPTER_REGISTRY[key] == TestAdapter

  # Verify get_adapter instantiates
  instance = get_adapter(key)
  assert isinstance(instance, TestAdapter)
  assert instance.initialized is True


def test_get_adapter_missing():
  """Verify returns None for unknown keys."""
  assert get_adapter("non_existent_framework") is None


# --- Snapshot Loading Tests ---


@pytest.fixture
def mock_snapshot_dir(tmp_path):
  """Create a fake snapshot directory."""
  d = tmp_path / "snapshots"
  d.mkdir()
  return d


def test_load_snapshot_sorts_versions(mock_snapshot_dir):
  """
  Verify logical sorting of versions (lexical sort by filename).
  """
  # Create files
  (mock_snapshot_dir / "testfw_v1.0.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
  (mock_snapshot_dir / "testfw_v2.0.json").write_text(json.dumps({"version": "2.0"}), encoding="utf-8")
  (mock_snapshot_dir / "testfw_v1.5.json").write_text(json.dumps({"version": "1.5"}), encoding="utf-8")

  # Patch the module-level constant SNAPSHOT_DIR in base
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir):
    result = load_snapshot_for_adapter("testfw")

    # Lexical sort implies v2.0 > v1.5 > v1.0
  assert result["version"] == "2.0"


def test_load_snapshot_missing_dir():
  """Verify safe return of empty dict if global dir missing."""
  # Patch with non-existent path
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", Path("/non/existent")):
    result = load_snapshot_for_adapter("any")
    assert result == {}


def test_load_snapshot_no_match(mock_snapshot_dir):
  """Verify safe return if directory exists but no file matches."""
  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir):
    result = load_snapshot_for_adapter("other_fw")
    assert result == {}


def test_load_snapshot_corrupt_file(mock_snapshot_dir, caplog):
  """Verify error handling on JSON decode error."""
  (mock_snapshot_dir / "corrupt_v1.json").write_text("{bad json", encoding="utf-8")

  with patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", mock_snapshot_dir):
    with caplog.at_level(logging.ERROR):
      result = load_snapshot_for_adapter("corrupt")

      assert result == {}
      assert "Failed to load snapshot" in caplog.text
