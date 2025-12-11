"""
Integration Tests for DataLoader Plugin Wiring (Updated for Distributed Semantics).
"""

import json
import sys
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


def test_generation_and_execution_flow(tmp_path):
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"

  empty_mgr = SemanticsManager()
  empty_mgr._reverse_index = {}
  empty_mgr.data = {}

  scaffolder = Scaffolder(semantics=empty_mgr)
  scaffolder.inspector.inspect = MagicMock(return_value={})

  with patch("ml_switcheroo.discovery.scaffolder.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.discovery.scaffolder.resolve_snapshots_dir", return_value=snap_dir):
      # Target temp dir with scaffold
      scaffolder.scaffold(["torch"], sem_dir)

  # Verify spec creation in semantics
  extra_spec = sem_dir / "k_framework_extras.json"
  assert extra_spec.exists()
  spec_data = json.loads(extra_spec.read_text())
  assert "DataLoader" in spec_data
  assert "variants" not in spec_data["DataLoader"]

  # Verify JAX Mapping in snapshots
  jax_map = snap_dir / "jax_mappings.json"
  assert jax_map.exists()
  jax_data = json.loads(jax_map.read_text())

  # Check plugin wiring
  assert "DataLoader" in jax_data["mappings"]
  assert jax_data["mappings"]["DataLoader"]["requires_plugin"] == "convert_dataloader"
