"""
Tests for the CLI 'sync' command (FrameworkSyncer Integration).

Verifies that the command correctly:
1. Reads Abstract Specs from semantics/
2. Loads existing snapshots (overlays) from snapshots/
3. Invokes FrameworkSyncer to find implementations
4. Writes updated mappings to snapshots/
"""

import json
from unittest.mock import patch
import pytest
from ml_switcheroo.cli.handlers.snapshots import handle_sync


@pytest.fixture
def sync_env(tmp_path):
  """Creates a mock environment with semantics and snapshots dirs."""
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()

  # Create a dummy Spec
  spec_data = {"Abs": {"std_args": ["x"], "variants": {}}, "Add": {"std_args": ["a", "b"], "variants": {}}}
  (sem_dir / "k_array_api.json").write_text(json.dumps(spec_data))

  return sem_dir, snap_dir


@pytest.fixture
def mock_syncer():
  """Mocks FrameworkSyncer to avoid real imports."""
  # FIX: Patch in snapshots handler
  with patch("ml_switcheroo.cli.handlers.snapshots.FrameworkSyncer") as mock:
    instance = mock.return_value

    def side_effect(tier_data, fw):
      # Simulate finding "Abs" and "Add"
      if "Abs" in tier_data:
        if "variants" not in tier_data["Abs"]:
          tier_data["Abs"]["variants"] = {}
        # Simulate finding implementation
        tier_data["Abs"]["variants"][fw] = {"api": f"{fw}.abs"}

    instance.sync.side_effect = side_effect
    yield instance


def test_sync_creates_new_snapshot(sync_env, mock_syncer):
  """
  Scenario: No existing snapshot.
  Expectation: snapshots/mockfw_vlatest_map.json is created with found items.
  """
  sem_dir, snap_dir = sync_env

  # FIX: Patch paths in snapshots handler
  with patch("ml_switcheroo.cli.handlers.snapshots.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.cli.handlers.snapshots.resolve_snapshots_dir", return_value=snap_dir):
      # Patch version to force 'latest' instead of 'unknown'
      with patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="latest"):
        ret = handle_sync("mockfw")

  assert ret == 0

  snap_file = snap_dir / "mockfw_vlatest_map.json"
  assert snap_file.exists()

  data = json.loads(snap_file.read_text())
  assert data["__framework__"] == "mockfw"
  assert "Abs" in data["mappings"]
  assert data["mappings"]["Abs"]["api"] == "mockfw.abs"


def test_sync_updates_existing_snapshot(sync_env, mock_syncer):
  """
  Scenario: Snapshot exists with Manual override.
  Expectation: Manual override preserved (if syncer respects it), new items added.
  """
  sem_dir, snap_dir = sync_env

  # Create existing snapshot with override for Add
  existing = {"__framework__": "mockfw", "mappings": {"Add": {"api": "mockfw.manual_add"}}}
  (snap_dir / "mockfw_vlatest_map.json").write_text(json.dumps(existing))

  with patch("ml_switcheroo.cli.handlers.snapshots.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.cli.handlers.snapshots.resolve_snapshots_dir", return_value=snap_dir):
      with patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="latest"):
        handle_sync("mockfw")

  data = json.loads((snap_dir / "mockfw_vlatest_map.json").read_text())

  # Abs should be added (found by mock syncer)
  assert "Abs" in data["mappings"]
  assert data["mappings"]["Abs"]["api"] == "mockfw.abs"

  # Add should be preserved
  assert "Add" in data["mappings"]
  assert data["mappings"]["Add"]["api"] == "mockfw.manual_add"


def test_sync_handles_unknown_tier_files_gracefully(tmp_path):
  """
  Scenario: semantics dir is empty.
  Expectation: Command runs, finds nothing, logs info, exit 0.
  """
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()

  # FIX: Patch paths in snapshots handler
  with patch("ml_switcheroo.cli.handlers.snapshots.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.cli.handlers.snapshots.resolve_snapshots_dir", return_value=snap_dir):
      # Mock syncer to do nothing
      with patch("ml_switcheroo.cli.handlers.snapshots.FrameworkSyncer"):
        ret = handle_sync("ghostfw")

  assert ret == 0
  # No snapshot created
  assert not (snap_dir / "ghostfw_vlatest_map.json").exists()
