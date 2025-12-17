"""
Tests for Keras Adapter Examples.

Ensures that tiered examples provided by the KerasAdapter are syntactically valid
and contain expected Keras 3.0 idiomatic constructs.
"""

import ast
import pytest
from ml_switcheroo.frameworks.keras import KerasAdapter


@pytest.fixture
def adapter():
  return KerasAdapter()


def test_keras_examples_structure(adapter):
  """Verify the dictionary structure of bundled examples."""
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter):
  """Verify Math example uses keras.ops."""
  code = adapter.get_tiered_examples()["tier1_math"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check imports
  assert "from keras import ops" in code

  # 3. Check usage
  assert "ops.abs" in code
  assert "ops.add" in code
  assert "ops.mean" in code


def test_tier2_neural_validity(adapter):
  """Verify Neural example uses Functional API."""
  code = adapter.get_tiered_examples()["tier2_neural"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check Functional API elements
  assert "keras.Input" in code
  assert "layers.Conv2D" in code
  assert "layers.Flatten" in code
  assert "layers.Dense" in code
  assert "keras.Model" in code

  # 3. Ensure we construct model with (inputs, outputs)
  assert "keras.Model(inputs, outputs)" in code


def test_tier3_extras_validity(adapter):
  """Verify Extras example uses Keras Random SeedGenerator."""
  code = adapter.get_tiered_examples()["tier3_extras"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check SeedGenerator usage (Keras 3 specific)
  assert "random.SeedGenerator" in code
  assert "random.normal" in code
