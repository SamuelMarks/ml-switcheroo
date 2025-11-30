import json

from ml_switcheroo.discovery.scaffolder import Scaffolder


class MockInspector:
  def inspect(self, fw):
    if fw == "torch":
      return {
        "torch.abs": {"name": "abs", "params": ["x"], "is_function": True},
        "torch.nn.Linear": {"name": "Linear", "params": ["in", "out"], "is_class": True},
      }
    if fw == "jax":
      return {
        "jax.numpy.abs": {"name": "abs", "params": ["a"], "is_function": True}
        # JAX doesn't have Linear, so it shouldn't match Tier B automatically
      }
    return {}


def test_scaffolder_logic(tmp_path):
  scaffolder = Scaffolder()
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
    # JAX variant should be missing or empty because "Linear" != "Dense"
    assert "jax" not in data["Linear"]["variants"]
