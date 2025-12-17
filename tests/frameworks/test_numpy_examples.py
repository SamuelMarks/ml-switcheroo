"""
Tests for NumPy Adapter Examples.

Ensures that tiered examples provided by the NumpyAdapter are syntactically valid
and contain expected numpy patterns.
"""

import ast
import pytest
from ml_switcheroo.frameworks.numpy import NumpyAdapter


@pytest.fixture
def adapter():
  return NumpyAdapter()


def test_numpy_examples_structure(adapter):
  """Verify the dictionary structure of bundled examples."""
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter):
  """Verify Math example uses standard numpy ops."""
  code = adapter.get_tiered_examples()["tier1_math"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check imports
  assert "import numpy as np" in code

  # 3. Check usage
  assert "np.matmul" in code
  assert "np.abs" in code
  assert "np.linalg.norm" in code


def test_tier2_neural_validity(adapter):
  """Verify Neural example is a valid Python comment stub."""
  code = adapter.get_tiered_examples()["tier2_neural"]

  # 1. Check syntax (comments are valid python)
  ast.parse(code)

  # 2. Check stub message
  assert "Out of Scope" in code
  assert "NumPy" in code


def test_tier3_extras_validity(adapter):
  """Verify Extras example uses save/load."""
  code = adapter.get_tiered_examples()["tier3_extras"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check IO usage
  assert "np.save" in code
  assert "np.load" in code
  assert "file=" in code
