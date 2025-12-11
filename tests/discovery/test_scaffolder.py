import json
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInspector:
  def inspect(self, fw):
    if fw == "torch":
      return {"torch.abs": {"name": "abs", "type": "function", "params": ["x"]}}
    if fw == "jax":
      return {"jax.numpy.abs": {"name": "abs", "type": "function", "params": ["a"]}}
    return {}


def test_scaffolder_logic(tmp_path):
  clean_semantics = SemanticsManager()
  clean_semantics.data = {}
  clean_semantics._key_origins = {}

  scaffolder = Scaffolder(semantics=clean_semantics)
  scaffolder.inspector = MockInspector()

  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()

  with patch("ml_switcheroo.discovery.scaffolder.resolve_snapshots_dir", return_value=snap_dir):
    scaffolder.scaffold(["torch", "jax"], sem_dir)

    # Verify Spec (Semantics Directory)
  tier_a = sem_dir / "k_array_api.json"
  assert tier_a.exists()
  spec_data = json.loads(tier_a.read_text())
  assert "abs" in spec_data
  assert "variants" not in spec_data["abs"]

  # Verify Mapping (Snapshots Directory)
  jax_map = snap_dir / "jax_mappings.json"
  assert jax_map.exists()
  jax_data = json.loads(jax_map.read_text())

  assert "abs" in jax_data["mappings"]
  assert jax_data["mappings"]["abs"]["api"] == "jax.numpy.abs"
