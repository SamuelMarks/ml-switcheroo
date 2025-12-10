"""
Tests for Scaffolder Dtype support.
"""

import json
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


# Mock Inspector returning Attributes
class MockAttributeInspector:
  def inspect(self, fw):
    if fw == "torch":
      return {
        "torch.float32": {"name": "float32", "type": "attribute", "params": []},
        "torch.abs": {"name": "abs", "type": "function", "params": ["x"]},
      }
    if fw == "jax":
      # Match
      return {
        "jax.numpy.float32": {"name": "float32", "type": "attribute", "params": []},
        "jax.numpy.abs": {"name": "abs", "type": "function", "params": ["a"]},
      }
    return {}


def test_scaffolder_propagates_type_field(tmp_path):
  """
  Ensure the JSON output includes "type": "attribute" or "function".
  """
  clean_semantics = SemanticsManager()
  clean_semantics.data = {}
  clean_semantics._key_origins = {}

  scaffolder = Scaffolder(semantics=clean_semantics)
  scaffolder.inspector = MockAttributeInspector()

  scaffolder.scaffold(["torch", "jax"], tmp_path)

  outfile = tmp_path / "k_array_api.json"
  data = json.loads(outfile.read_text())

  # Check Attribute
  assert "float32" in data
  assert data["float32"]["type"] == "attribute"
  assert data["float32"]["variants"]["torch"]["api"] == "torch.float32"
  assert data["float32"]["variants"]["jax"]["api"] == "jax.numpy.float32"

  # Check Function
  assert "abs" in data
  assert data["abs"]["type"] == "function"
