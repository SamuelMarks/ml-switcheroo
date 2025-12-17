"""
Tests for TensorFlow Adapter Examples.

Ensures that tiered examples provided by the TensorFlowAdapter are syntactically valid
and contain expected TF Core patterns.
"""

import ast
import pytest
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter


@pytest.fixture
def adapter():
  return TensorFlowAdapter()


def test_tensorflow_examples_structure(adapter):
  """Verify the dictionary structure of bundled examples."""
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter):
  """Verify Math example uses tf.math."""
  code = adapter.get_tiered_examples()["tier1_math"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check imports
  assert "import tensorflow as tf" in code

  # 3. Check usage
  assert "tf.abs" in code
  assert "tf.math.add" in code
  assert "tf.math.reduce_mean" in code


def test_tier2_neural_validity(adapter):
  """Verify Neural example uses tf.Module and variables."""
  code = adapter.get_tiered_examples()["tier2_neural"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check low-level structure
  assert "class Model(tf.Module):" in code
  assert "tf.Variable" in code
  assert "tf.matmul" in code

  # 3. Ensure __call__ is used (standard for tf.Module)
  assert "def __call__(self, x):" in code


def test_tier3_extras_validity(adapter):
  """Verify Extras example uses tf.data.Dataset."""
  code = adapter.get_tiered_examples()["tier3_extras"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check dataset operations
  assert "tf.data.Dataset.from_tensor_slices" in code
  assert ".shuffle" in code
  assert ".batch" in code
