"""Test suite for the Dataloader Wiring module."""

import json
import typing
from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.semantics.manager import SemanticsManager


def test_generation_and_execution_flow(tmp_path: Path) -> None:
  """Verifies the behavior of generation and execution flow."""
  sem_dir: Path = tmp_path / "semantics"
  snap_dir: Path = tmp_path / "snapshots"
  sem_dir.mkdir(parents=True)
  snap_dir.mkdir(parents=True)
  jax_snapshot_content: dict[str, typing.Any] = {
    "__framework__": "jax",
    "mappings": {"DataLoader": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"}},
  }
  (snap_dir / "jax_vlatest_map.json").write_text(json.dumps(jax_snapshot_content), encoding="utf-8")
  extras_content: dict[str, typing.Any] = {"DataLoader": {"std_args": ["dataset"], "description": "Load Dataset"}}
  import yaml

  odl_dir: Path = sem_dir / "odl"
  odl_dir.mkdir()
  (odl_dir / "DataLoader.yaml").write_text(yaml.dump(extras_content), encoding="utf-8")
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap_dir):
      SemanticsManager()
  extra_spec: Path = sem_dir / "odl" / "DataLoader.yaml"
  assert extra_spec.exists()
  import yaml

  spec_data: typing.Any = yaml.safe_load(extra_spec.read_text())
  assert "DataLoader" in spec_data
  assert "variants" not in spec_data["DataLoader"]
  assert "std_args" in spec_data["DataLoader"]
  jax_map: Path = snap_dir / "jax_vlatest_map.json"
  assert jax_map.exists()
  jax_data: typing.Any = json.loads(jax_map.read_text())
  assert "DataLoader" in jax_data["mappings"]
  assert jax_data["mappings"]["DataLoader"]["requires_plugin"] == "convert_dataloader"
  assert jax_data["mappings"]["DataLoader"]["api"] == "GenericDataLoader"
