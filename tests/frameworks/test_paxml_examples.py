"""
Tests for PaxML Adapter Examples.

Ensures that tiered examples provided by the PaxmlAdapter are syntactically valid
and contain expected Praxis patterns (setup() lifecycle).
"""

import ast
import pytest
from ml_switcheroo.frameworks.paxml import PaxmlAdapter


@pytest.fixture
def adapter():
  return PaxmlAdapter()


def test_paxml_examples_structure(adapter):
  """Verify the dictionary structure of bundled examples."""
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter):
  """Verify Math example delegates to JAX Core."""
  code = adapter.get_tiered_examples()["tier1_math"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check JAX inheritance
  assert "import jax.numpy as jnp" in code


def test_tier2_neural_validity(adapter):
  """Verify Neural example uses Praxis lifecycle (setup vs init)."""
  code = adapter.get_tiered_examples()["tier2_neural"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check Base Layer
  assert "class SimpleMLP(base_layer.BaseLayer):" in code

  # 3. Check Lifecycle Method (setup not init)
  assert "def setup(self):" in code
  assert "pl.Linear" in code


def test_tier3_extras_validity(adapter):
  """Verify Extras example uses HParams/Fiddle."""
  code = adapter.get_tiered_examples()["tier3_extras"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check Config logic
  assert "pax_fiddle.Config" in code
  assert "input_dims" in code
