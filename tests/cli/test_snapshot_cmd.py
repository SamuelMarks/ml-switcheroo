"""
Tests for the CLI 'snapshot' command.

Verifies that the command correctly iterates installed adapters,
calls collect_api, and writes the JSON output to the specified folder.
"""

import pytest
import json
from unittest.mock import patch, MagicMock

from ml_switcheroo.cli.handlers.snapshots import handle_snapshot
from ml_switcheroo.frameworks.base import GhostRef

# --- Fixtures & Mocks ---


@pytest.fixture
def mock_snapshot_dir(tmp_path):
  return tmp_path


def mock_get_adapter(name):
  """Returns a fake adapter for different frameworks."""
  adapter = MagicMock()

  # Mock collect_api behavior
  if name == "torch":
    # Returns one ghost ref
    ref = GhostRef(name="MSELoss", api_path="torch.nn.MSELoss", kind="class", params=[])
    adapter.collect_api.return_value = [ref]

  elif name == "keras":
    # Simulates uninstalled lib behavior if we used old script logic,
    # but here we mock the adapter object itself.
    # Let's say it returns nothing to test skipping.
    adapter.collect_api.return_value = []

  return adapter


def mock_get_pkg_version(name):
  if name == "torch":
    return "1.0.0"
  if name == "keras":
    return "unknown"
  return "0.0.1"


# --- Tests ---


# FIX: Patch handlers.snapshots where the code lives
@patch("ml_switcheroo.cli.handlers.snapshots.available_frameworks", return_value=["torch", "keras"])
@patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", side_effect=mock_get_adapter)
@patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", side_effect=mock_get_pkg_version)
def test_snapshot_command_flow(mock_ver, mock_get_adp, mock_avail, mock_snapshot_dir):
  """
  Verify standard execution flow:
  1. Iterate frameworks.
  2. Check version (skip unknown).
  3. Collect API.
  4. Write File.
  """
  # Run command targeting temp dir
  ret_code = handle_snapshot(out_dir=mock_snapshot_dir)

  assert ret_code == 0

  # Check outputs
  # 1. Torch should exist (version 1.0.0 is valid, collect_api returned data)
  torch_file = mock_snapshot_dir / "torch_v1.0.0.json"
  assert torch_file.exists()

  with open(torch_file, "r") as f:
    data = json.load(f)
    assert data["version"] == "1.0.0"
    assert len(data["categories"]) > 0

  # 2. Keras should NOT exist (version unknown logic inside _capture_framework)
  keras_file = mock_snapshot_dir / "keras_vunknown.json"
  assert not keras_file.exists()


@patch("ml_switcheroo.cli.handlers.snapshots.available_frameworks", return_value=[])
def test_snapshot_no_frameworks(mock_avail):
  """Verify error code when no frameworks registry found."""
  ret = handle_snapshot(out_dir=None)
  assert ret == 1
