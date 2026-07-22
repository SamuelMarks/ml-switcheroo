"""Test suite for the Dataloader Wiring module."""

import json
from unittest.mock import patch
from ml_switcheroo.semantics.manager import SemanticsManager


def test_generation_and_execution_flow(tmp_path):
  """Verifies the behavior of generation and execution flow."""
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir(parents=True)
  snap_dir.mkdir(parents=True)
  jax_snapshot_content = {
    "__framework__": "jax",
    "mappings": {"DataLoader": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"}},
  }
  (snap_dir / "jax_vlatest_map.json").write_text(json.dumps(jax_snapshot_content), encoding="utf-8")
  extras_content = {"DataLoader": {"std_args": ["dataset"], "description": "Load Dataset"}}
  import yaml

  odl_dir = sem_dir / "odl"
  odl_dir.mkdir()
  (odl_dir / "DataLoader.yaml").write_text(yaml.dump(extras_content), encoding="utf-8")
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap_dir):
      SemanticsManager()
  extra_spec = sem_dir / "odl" / "DataLoader.yaml"
  assert extra_spec.exists()
  import yaml

  spec_data = yaml.safe_load(extra_spec.read_text())
  assert "DataLoader" in spec_data
  assert "variants" not in spec_data["DataLoader"]
  assert "std_args" in spec_data["DataLoader"]
  jax_map = snap_dir / "jax_vlatest_map.json"
  assert jax_map.exists()
  jax_data = json.loads(jax_map.read_text())
  assert "DataLoader" in jax_data["mappings"]
  assert jax_data["mappings"]["DataLoader"]["requires_plugin"] == "convert_dataloader"
  assert jax_data["mappings"]["DataLoader"]["api"] == "GenericDataLoader"
