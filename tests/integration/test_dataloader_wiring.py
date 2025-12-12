"""
Integration Tests for DataLoader Plugin Wiring (Updated for Distributed Semantics).

Verifies that:
1. Scaffolder injects Abstract Definitions from Python code into semantics JSON.
2. The system correctly respects Implementation Variants provided in existing snapshots.
"""

import json
import sys
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.scaffolder import Scaffolder
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

  # 2. Setup Scaffolder
  empty_mgr = SemanticsManager()
  # Bypass internal loading logic to ensure test isolation
  empty_mgr._reverse_index = {}
  empty_mgr.data = {}
  # Note: Scaffolder calls get_dataloader_semantics() internally to populate Abstract Specs

  scaffolder = Scaffolder(semantics=empty_mgr)
  scaffolder.inspector.inspect = MagicMock(return_value={})

  # 3. Execution
  # Pass tmp_path as root
  scaffolder.scaffold(["torch"], root_dir=tmp_path)

  # 4. Verify Abstract Spec Creation (semantics/k_framework_extras.json)
  # This comes from the python file (common/data.py)
  extra_spec = sem_dir / "k_framework_extras.json"
  assert extra_spec.exists()
  spec_data = json.loads(extra_spec.read_text())

  assert "DataLoader" in spec_data
  # Key Verification: Variants must NOT be in the Abstract Spec
  assert "variants" not in spec_data["DataLoader"]
  assert "std_args" in spec_data["DataLoader"]

  # 5. Verify Implementation Persistence (snapshots/jax_vlatest_map.json)
  # Ensure the pre-seeded mapping was preserved
  jax_map = snap_dir / "jax_vlatest_map.json"
  assert jax_map.exists()
  jax_data = json.loads(jax_map.read_text())

  # Check plugin wiring matches the snapshot we seeded
  assert "DataLoader" in jax_data["mappings"]
  assert jax_data["mappings"]["DataLoader"]["requires_plugin"] == "convert_dataloader"
  assert jax_data["mappings"]["DataLoader"]["api"] == "GenericDataLoader"
