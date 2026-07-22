"""Test suite for the Tensorflow Examples module."""

import ast
import pytest
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter


@pytest.fixture
def adapter():
  """Provides a mock adapter for testing."""
  return TensorFlowAdapter()


def test_tensorflow_examples_structure(adapter):
  """Verifies the behavior of TensorFlow examples structure."""
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter):
  """Verifies the behavior of tier1 math validity."""
  code = adapter.get_tiered_examples()["tier1_math"]
  ast.parse(code)
  assert "import tensorflow as tf" in code
  assert "tf.abs" in code
  assert "tf.math.add" in code
  assert "tf.math.reduce_mean" in code


def test_tier2_neural_validity(adapter):
  """Verifies the behavior of tier2 neural validity."""
  code = adapter.get_tiered_examples()["tier2_neural"]
  ast.parse(code)
  assert "class Model(tf.Module):" in code
  assert "tf.Variable" in code
  assert "tf.matmul" in code
  assert "def __call__(self, x):" in code


def test_tier3_extras_validity(adapter):
  """Verifies the behavior of tier3 extras validity."""
  code = adapter.get_tiered_examples()["tier3_extras"]
  ast.parse(code)
  assert "tf.data.Dataset.from_tensor_slices" in code
  assert ".shuffle" in code
  assert ".batch" in code
