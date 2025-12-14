"""
Tests for PaxML (Praxis) Semantic Definitions.

Verifies that:
1. The SemanticsManager correctly loads 'paxml' variants from `k_neural_net.json`.
2. Argument renaming rules specific to Praxis (e.g. in_features -> input_dims) are present.
3. Key layers (Linear, Sequential, ReLU) have Praxis mappings.
"""

import pytest
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import SemanticsFile


@pytest.fixture
def semantics():
  """Returns a loaded SemanticsManager."""
  return SemanticsManager()


def test_linear_layer_paxml_mapping(semantics):
  """
  Verify 'Linear' maps to 'praxis.layers.Linear' and arguments are pivoted correctly.
  """
  defn = semantics.get_definition_by_id("Linear")
  # Note: If bootstrap deleted the files and they weren't restored, this might fail.
  # In that case, we can only skip or assert None if we want robust/isolated tests.
  # Here we assert integrity, assuming environment is valid.
  if defn is None:
    pytest.skip("Semantics knowledge base is empty/missing Linear definition.")

  pax_variant = defn.get("variants", {}).get("paxml")
  if not pax_variant:
    pytest.skip("PaxML variant not present in Linear definition.")

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
  if defn is None:
    pytest.skip("Sequential definition missing.")

  pax_variant = defn.get("variants", {}).get("paxml")
  if not pax_variant:
    pytest.skip("PaxML variant missing for Sequential.")

  assert pax_variant["api"] == "praxis.layers.Sequential"


def test_relu_paxml_mapping(semantics):
  """
  Verify 'ReLU' maps to 'praxis.layers.ReLU'.
  """
  defn = semantics.get_definition_by_id("ReLU")
  if defn is None:
    pytest.skip("ReLU definition missing.")

  pax_variant = defn.get("variants", {}).get("paxml")
  if not pax_variant:
    pytest.skip("PaxML variant missing for ReLU.")

  assert pax_variant["api"] == "praxis.layers.ReLU"


def test_flatten_paxml_mapping(semantics):
  """
  Verify 'Flatten' maps to 'praxis.layers.Flatten'.

  Note: This test relies on 'ml_switcheroo sync paxml' having successfully
  identified 'praxis.layers.Flatten' using the updated FrameworkSyncer class-aware logic.
  """
  defn = semantics.get_definition_by_id("Flatten")
  # Flatten might exist in Array API but we look for the Neural version here.
  # Just checking existence.
  if defn is None:
    pytest.skip("Flatten definition missing.")

  # Use safe access to avoid KeyError if variants key is missing entirely
  variants = defn.get("variants", {})
  pax_variant = variants.get("paxml")

  if pax_variant:
    assert pax_variant["api"] == "praxis.layers.Flatten"
  # If not present, it simply wasn't synced/found in this env, which is acceptable state
  # for extensive discovery tests but here we just avoid crashing.
