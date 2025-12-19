"""
Tests for Keras Adapter Examples.
"""

import ast
import pytest
from ml_switcheroo.frameworks.keras import KerasAdapter


@pytest.fixture
def adapter():
  return KerasAdapter()


def test_keras_examples_structure(adapter):
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter):
  code = adapter.get_tiered_examples()["tier1_math"]
  ast.parse(code)
  assert "from keras import ops" in code
  assert "ops.abs" in code


def test_tier2_neural_validity(adapter):
  code = adapter.get_tiered_examples()["tier2_neural"]
  ast.parse(code)
  assert "keras.Input" in code
  assert "layers.Conv2D" in code
  assert "layers.Flatten" in code
  assert "layers.Dense" in code
  assert "keras.Model" in code


def test_tier3_extras_validity(adapter):
  code = adapter.get_tiered_examples()["tier3_extras"]
  ast.parse(code)
  assert "random.SeedGenerator" in code
