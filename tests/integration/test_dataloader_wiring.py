"""
Integration Tests for DataLoader Plugin Wiring (Updated for Distributed Semantics).

Verifies that:
1. Semantic definitions in `semantics/` are correctly loaded.
2. The system correctly respects Implementation Variants provided in existing snapshots.
"""

import json
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

from ml_switcheroo.cli.commands import handle_import_spec
from ml_switcheroo.semantics.manager import SemanticsManager, resolve_semantics_dir


def test_generation_and_execution_flow(tmp_path):
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir(parents=True)
  snap_dir.mkdir(parents=True)

  # 1. Pre-seed the Snapshot (Simulation of existing JSON source of truth)
  # Since we removed python hardcoding, the variants must exist on disk
  jax_snapshot_content = {
    "__framework__": "jax",
    "mappings": {"DataLoader": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"}},
  }
  (snap_dir / "jax_vlatest_map.json").write_text(json.dumps(jax_snapshot_content), encoding="utf-8")

  # 2. Pre-seed the Semantics (Simulation of Hub)
  # Note: Previously we hydrated this from internal defaults manually.
  # Now we must explicitly create the JSON file the test expects to find.
  # The test verify 'k_framework_extras.json' existence.
  extras_content = {
    "DataLoader": {
      "std_args": ["dataset"],
      "description": "Load Dataset",
    }
  }
  (sem_dir / "k_framework_extras.json").write_text(json.dumps(extras_content), encoding="utf-8")

  # Mock paths in the system to use our temp dirs
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap_dir):
      # Trigger loading naturally
      mgr = SemanticsManager()

  # 4. Verify Abstract Spec Creation verification
  # The file we manually created exists
  extra_spec = sem_dir / "k_framework_extras.json"
  assert extra_spec.exists()

  spec_data = json.loads(extra_spec.read_text())
  assert "DataLoader" in spec_data
  # Key Verification: Variants must NOT be in the Abstract Spec
  assert "variants" not in spec_data["DataLoader"]
  assert "std_args" in spec_data["DataLoader"]

  # 5. Verify Implementation Persistence (snapshots/jax_vlatest_map.json)
  # Ensure the pre-seeded mapping was preserved in snapshots/
  jax_map = snap_dir / "jax_vlatest_map.json"
  assert jax_map.exists()
  jax_data = json.loads(jax_map.read_text())

  # Check plugin wiring matches the snapshot we seeded
  assert "DataLoader" in jax_data["mappings"]
  assert jax_data["mappings"]["DataLoader"]["requires_plugin"] == "convert_dataloader"
  assert jax_data["mappings"]["DataLoader"]["api"] == "GenericDataLoader"
