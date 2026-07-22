"""Test suite for the Paxml Examples module."""

import ast
import pytest
from ml_switcheroo.frameworks.paxml import PaxmlAdapter


@pytest.fixture
def adapter():
  """Provides a mock adapter for testing."""
  return PaxmlAdapter()


def test_paxml_examples_structure(adapter):
  """Verifies the behavior of Paxml examples structure."""
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter):
  """Verifies the behavior of tier1 math validity."""
  code = adapter.get_tiered_examples()["tier1_math"]
  ast.parse(code)
  assert "import jax.numpy as jnp" in code


def test_tier2_neural_validity(adapter):
  """Verifies the behavior of tier2 neural validity."""
  code = adapter.get_tiered_examples()["tier2_neural"]
  ast.parse(code)
  assert "class SimpleMLP(base_layer.BaseLayer):" in code
  assert "def setup(self):" in code
  assert "pl.Linear" in code


def test_tier3_extras_validity(adapter):
  """Verifies the behavior of tier3 extras validity."""
  code = adapter.get_tiered_examples()["tier3_extras"]
  ast.parse(code)
  assert "pax_fiddle.Config" in code
  assert "input_dims" in code
