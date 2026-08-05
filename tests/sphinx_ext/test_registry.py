"""Test suite for the Registry module."""

from unittest.mock import patch, MagicMock
from ml_switcheroo.sphinx_ext.registry import scan_registry
import json


@patch("ml_switcheroo.sphinx_ext.registry.available_frameworks")
@patch("ml_switcheroo.sphinx_ext.registry.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.registry.get_adapter")
def test_scan_registry_else_fallback(mock_get_adapter, mock_priority, mock_avail):
  """Tests the else fallback when no other candidates are in the priority list."""
  mock_avail.return_value = ["torch", "custom1", "custom2"]
  mock_priority.return_value = ["torch", "custom1"]

  def get_adapter_side_effect(name):
    """Effect."""
    adapter = MagicMock()
    adapter.inherits_from = None
    adapter.display_name = name.title()
    adapter.supported_tiers = None
    adapter.get_tiered_examples.return_value = {"example1": f"{name}_code"}
    return adapter

  mock_get_adapter.side_effect = get_adapter_side_effect
  (hierarchy, examples_json, tier_metadata_json) = scan_registry()
  examples = json.loads(examples_json)
  # torch will fall back to custom1 (first candidate)
  assert examples["torch_example1"]["tgtFw"] == "custom1"


@patch("ml_switcheroo.sphinx_ext.registry.available_frameworks")
@patch("ml_switcheroo.sphinx_ext.registry.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.registry.get_adapter")
def test_scan_registry(mock_get_adapter, mock_priority, mock_avail):
  """Scans registry."""
  mock_avail.return_value = ["torch", "jax", "flax_nnx", "unknown"]
  mock_priority.return_value = ["torch", "jax"]

  def get_adapter_side_effect(name):
    """Effect."""
    if name == "torch":
      adapter = MagicMock()
      adapter.inherits_from = None
      adapter.display_name = "PyTorch"
      adapter.supported_tiers = None
      adapter.get_tiered_examples.return_value = {"tier1_math": "torch_math"}
      return adapter
    elif name == "jax":
      adapter = MagicMock()
      adapter.inherits_from = None
      adapter.display_name = "JAX"
      tier = MagicMock()
      tier.value = "array"
      adapter.supported_tiers = [tier]
      adapter.get_tiered_examples.return_value = {"tier2_neural": "jax_nn"}
      return adapter
    elif name == "flax_nnx":
      adapter = MagicMock()
      adapter.inherits_from = "jax"
      adapter.display_name = "Flax"
      adapter.supported_tiers = None
      adapter.get_tiered_examples.return_value = {"tier1_math": "flax_math"}
      return adapter
    return None

  mock_get_adapter.side_effect = get_adapter_side_effect
  (hierarchy, examples_json, tier_metadata_json) = scan_registry()
  assert "torch" in hierarchy
  assert "jax" in hierarchy
  assert len(hierarchy["jax"]) == 1
  assert hierarchy["jax"][0]["key"] == "flax_nnx"
  examples = json.loads(examples_json)
  assert "torch_tier1_math" in examples
  assert "jax_tier2_neural" in examples
  assert "flax_nnx_tier1_math" in examples
  assert examples["torch_tier1_math"]["tgtFw"] == "jax"
  tier_metadata = json.loads(tier_metadata_json)
  assert "torch" in tier_metadata
  assert "array" in tier_metadata["jax"]
