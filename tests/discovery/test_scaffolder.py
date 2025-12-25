import json
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInspector:
  def inspect(self, fw):
    # Updated to handle search_modules iteration (e.g. 'jax.numpy')
    if "torch" in fw:
      return {"torch.abs": {"name": "abs", "type": "function", "params": ["x"]}}
    if "jax" in fw:
      return {"jax.numpy.abs": {"name": "abs", "type": "function", "params": ["a"]}}
    return {}


def test_scaffolder_logic(tmp_path):
  clean_semantics = SemanticsManager()
  clean_semantics.data = {}
  clean_semantics._key_origins = {}

  scaffolder = Scaffolder(semantics=clean_semantics)
  scaffolder.inspector = MockInspector()

  # Ensure get_adapter returns None so Scaffolder iterates ['torch'] directly
  # rather than attempting to resolve complex submodules against our simple MockInspector
  with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=None):
    scaffolder.scaffold(["torch", "jax"], root_dir=tmp_path)

  # Verify Spec (Semantics Directory created inside root)
  # Without pre-seeded knowledge, Scaffolder might route to extras.
  # Check extras if array api missing.

  tier_a = tmp_path / "semantics" / "k_array_api.json"
  tier_extras = tmp_path / "semantics" / "k_framework_extras.json"

  spec_data = {}
  if tier_a.exists():
    spec_data.update(json.loads(tier_a.read_text()))
  if tier_extras.exists():
    spec_data.update(json.loads(tier_extras.read_text()))

  assert "abs" in spec_data
  assert "variants" not in spec_data["abs"]

  # Verify Mapping (Snapshots Directory created inside root)
  snap_dir = tmp_path / "snapshots"
  assert snap_dir.exists()

  # Logic tries importlib version, fails -> "latest"
  jax_maps = list(snap_dir.glob("jax_v*_map.json"))
  assert len(jax_maps) > 0
  jax_map = jax_maps[0]

  jax_data = json.loads(jax_map.read_text())

  assert "abs" in jax_data["mappings"]
  assert jax_data["mappings"]["abs"]["api"] == "jax.numpy.abs"
