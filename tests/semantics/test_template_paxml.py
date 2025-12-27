"""
Tests for PaxML Template Configuration.
Updated to check snapshots/paxml_vlatest_map.json instead of k_test_templates.json.
"""

import json
from unittest.mock import patch
from ml_switcheroo.cli.handlers.snapshots import handle_sync
from ml_switcheroo.frameworks.paxml import PaxmlAdapter


def test_paxml_template_exists_on_disk(tmp_path):
  """
  Verify the mapping file exists and contains templates.
  We force a sync here to ensure the file populated in the test environment.
  """
  # 1. Setup Mock Environment
  snap_dir = tmp_path / "snapshots"
  sem_dir = tmp_path / "semantics"
  snap_dir.mkdir()
  sem_dir.mkdir()

  # 2. Patch paths and version
  with patch("ml_switcheroo.cli.handlers.snapshots.resolve_snapshots_dir", return_value=snap_dir):
    with patch("ml_switcheroo.cli.handlers.snapshots.resolve_semantics_dir", return_value=sem_dir):
      with patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="latest"):
        # 3. Helper to return full adapter
        with patch("ml_switcheroo.cli.handlers.snapshots.get_adapter") as mock_get:
          # Provide a real adapter instance so .test_config is accessible
          mock_get.return_value = PaxmlAdapter()

          # 4. Run Sync
          handle_sync("paxml")

  # 5. Verify
  map_file = snap_dir / "paxml_vlatest_map.json"
  assert map_file.exists(), "Snapshot file was not created"

  content = json.loads(map_file.read_text())
  assert "templates" in content, "Templates key missing from snapshot"

  pax_conf = content["templates"]
  assert "import" in pax_conf
  assert "praxis" in pax_conf["import"]
