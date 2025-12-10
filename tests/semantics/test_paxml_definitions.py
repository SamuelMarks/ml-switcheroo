"""
Tests for PaxML (Praxis) Semantic Definitions.

Verifies that:
1. The SemanticsManager correctly loads 'paxml' variants from `k_neural_net.json`.
2. Argument renaming rules specific to Praxis (e.g. in_features -> input_dims) are present.
3. Key layers (Linear, Sequential, ReLU) have Praxis mappings.
"""

import pytest
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def semantics():
  """Returns a loaded SemanticsManager."""
  return SemanticsManager()


def test_linear_layer_paxml_mapping(semantics):
  """
  Verify 'Linear' maps to 'praxis.layers.Linear' and arguments are pivoted correctly.
  """
  defn = semantics.get_definition_by_id("Linear")
  assert defn is not None, "Linear definition missing from Semantics"

  pax_variant = defn["variants"].get("paxml")
  assert pax_variant is not None, "PaxML variant missing for Linear"

  # Check API Path
  assert pax_variant["api"] == "praxis.layers.Linear"

  # Check Argument Renaming
  # Standard: in_features -> Praxis: input_dims
  # Standard: out_features -> Praxis: output_dims
  args_map = pax_variant.get("args", {})
  assert args_map["in_features"] == "input_dims"
  assert args_map["out_features"] == "output_dims"
  assert args_map["bias"] == "use_bias"


def test_sequential_container_paxml_mapping(semantics):
  """
  Verify 'Sequential' maps to 'praxis.layers.Sequential'.
  """
  defn = semantics.get_definition_by_id("Sequential")
  assert defn is not None

  pax_variant = defn["variants"].get("paxml")
  assert pax_variant is not None
  assert pax_variant["api"] == "praxis.layers.Sequential"


def test_relu_paxml_mapping(semantics):
  """
  Verify 'ReLU' maps to 'praxis.layers.ReLU'.
  """
  defn = semantics.get_definition_by_id("ReLU")
  assert defn is not None

  pax_variant = defn["variants"].get("paxml")
  assert pax_variant is not None
  assert pax_variant["api"] == "praxis.layers.ReLU"


def test_flatten_paxml_mapping(semantics):
  """
  Verify 'Flatten' maps to 'praxis.layers.Flatten'.

  Note: This test relies on 'ml_switcheroo sync paxml' having successfully
  identified 'praxis.layers.Flatten' using the updated FrameworkSyncer class-aware logic.
  """
  defn = semantics.get_definition_by_id("Flatten")
  assert defn is not None

  pax_variant = defn["variants"].get("paxml")

  # This assertion validates the syncer fix.
  # If syncer.py logic is correct, finding 'praxis.layers.Flatten' works,
  # and this test passes after a sync run.
  assert pax_variant is not None
  assert pax_variant["api"] == "praxis.layers.Flatten"
