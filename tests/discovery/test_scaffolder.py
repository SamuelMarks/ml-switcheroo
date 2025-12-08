import json

from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInspector:
  def inspect(self, fw):
    if fw == "torch":
      return {
        "torch.abs": {"name": "abs", "type": "function", "params": ["x"]},
        "torch.nn.Linear": {"name": "Linear", "type": "class", "params": ["in", "out"]},
      }
    if fw == "jax":
      return {
        "jax.numpy.abs": {"name": "abs", "type": "function", "params": ["a"]}
        # JAX doesn't have Linear class in this mock, so it shouldn't match Tier B automatically
      }
    return {}


def test_scaffolder_logic(tmp_path):
  """
  Tests that scaffolder identifies and sorts APIs into correct JSONs.
  Note: We inject a clean SemanticsManager to prevent real K_JSONs from polluting the test state.
  """
  # Inject clean semantics so we don't pick up 'Linear' from real k_neural_net.json
  clean_semantics = SemanticsManager()
  clean_semantics.data = {}
  clean_semantics._key_origins = {}
  # Avoid loading defaults

  scaffolder = Scaffolder(semantics=clean_semantics)
  scaffolder.inspector = MockInspector()

  # Run
  scaffolder.scaffold(["torch", "jax"], tmp_path)

  # Verify Tier A (Math)
  tier_a = tmp_path / "k_array_api.json"
  assert tier_a.exists()
  with open(tier_a, "rt", encoding="utf-8") as f:
    data = json.load(f)
    assert "abs" in data
    assert "torch" in data["abs"]["variants"]
    assert "jax" in data["abs"]["variants"]
    # It matched torch.abs to jax.numpy.abs via the name "abs"
    assert data["abs"]["variants"]["jax"]["api"] == "jax.numpy.abs"

    # Verify Tier B (Neural)
  tier_b = tmp_path / "k_neural_net.json"
  with open(tier_b, "rt", encoding="utf-8") as f:
    data = json.load(f)
    assert "Linear" in data
    assert "torch" in data["Linear"]["variants"]
    # JAX variant should be missing because our mock inspector for JAX
    # didn't return a 'Linear' match and clean_semantics didn't have one pre-defined.
    assert "jax" not in data["Linear"]["variants"]
