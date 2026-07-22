"""Test suite for the Keras Examples module."""

import ast
import pytest
from ml_switcheroo.frameworks.keras import KerasAdapter


@pytest.fixture
def adapter():
  """Provides a mock adapter for testing."""
  return KerasAdapter()


def test_keras_examples_structure(adapter):
  """Verifies the behavior of Keras examples structure."""
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural_sequential" in examples
  assert "tier3_extras_rng" in examples


def test_tier1_math_validity(adapter):
  """Verifies the behavior of tier1 math validity."""
  code = adapter.get_tiered_examples()["tier1_math"]
  ast.parse(code)
  assert "import keras" in code
  assert "keras.ops.abs" in code


def test_tier2_neural_validity(adapter):
  """Verifies the behavior of tier2 neural validity."""
  code = adapter.get_tiered_examples()["tier2_neural_sequential"]
  ast.parse(code)
  assert "keras.Sequential" in code


def test_tier3_extras_validity(adapter):
  """Verifies the behavior of tier3 extras validity."""
  code = adapter.get_tiered_examples()["tier3_extras_rng"]
  ast.parse(code)
  assert "random.SeedGenerator" in code
