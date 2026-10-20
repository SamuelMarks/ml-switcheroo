"""Test suite for the Paxml Definitions module."""

import typing

import pytest

from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def semantics() -> typing.Generator[SemanticsManager, None, None]:
  """Provides a mock semantics for testing.

  Yields:
      SemanticsManager: A semantics manager instance.
  """
  yield SemanticsManager()


def test_linear_layer_paxml_mapping(semantics: SemanticsManager) -> None:
  """Verifies the behavior of linear layer Paxml mapping.

  Args:
      semantics (SemanticsManager): The semantics manager fixture.
  """
  defn = semantics.get_definition_by_id("Linear")
  if defn is None:
    pytest.skip("Semantics knowledge base is empty/missing Linear definition.")
  variants = defn.get("variants")
  assert isinstance(variants, dict)
  pax_variant = variants.get("paxml")
  if not isinstance(pax_variant, dict):
    pytest.skip("PaxML variant not present in Linear definition.")
  assert pax_variant["api"] == "praxis.layers.Linear"
  args_map = pax_variant.get("args")
  assert isinstance(args_map, dict)
  assert args_map["in_features"] == "input_dims"
  assert args_map["out_features"] == "output_dims"
  assert args_map["bias"] == "use_bias"


def test_sequential_container_paxml_mapping(semantics: SemanticsManager) -> None:
  """Verifies the behavior of sequential container Paxml mapping.

  Args:
      semantics (SemanticsManager): The semantics manager fixture.
  """
  defn = semantics.get_definition_by_id("Sequential")
  if defn is None:
    pytest.skip("Sequential definition missing.")
  variants = defn.get("variants")
  assert isinstance(variants, dict)
  pax_variant = variants.get("paxml")
  if not isinstance(pax_variant, dict):
    pytest.skip("PaxML variant missing for Sequential.")
  assert pax_variant["api"] == "praxis.layers.Sequential"


def test_relu_paxml_mapping(semantics: SemanticsManager) -> None:
  """Verifies the behavior of relu Paxml mapping.

  Args:
      semantics (SemanticsManager): The semantics manager fixture.
  """
  defn = semantics.get_definition_by_id("ReLU")
  if defn is None:
    pytest.skip("ReLU definition missing.")
  variants = defn.get("variants")
  assert isinstance(variants, dict)
  pax_variant = variants.get("paxml")
  if not isinstance(pax_variant, dict):
    pytest.skip("PaxML variant missing for ReLU.")
  assert pax_variant["api"] == "praxis.layers.ReLU"


def test_flatten_paxml_mapping(semantics: SemanticsManager) -> None:
  """Verifies the behavior of flatten Paxml mapping.

  Args:
      semantics (SemanticsManager): The semantics manager fixture.
  """
  defn = semantics.get_definition_by_id("Flatten")
  if defn is None:
    pytest.skip("Flatten definition missing.")
  variants = defn.get("variants")
  assert isinstance(variants, dict)
  pax_variant = variants.get("paxml")
  if isinstance(pax_variant, dict):
    if "api" in pax_variant:
      assert pax_variant["api"] == "praxis.layers.Flatten"
    elif "requires_plugin" in pax_variant:
      assert pax_variant["requires_plugin"] == "flatten_range"
