"""Test suite for the Numpy Examples module."""

import ast
import pytest
from ml_switcheroo.frameworks.numpy import NumpyAdapter


@pytest.fixture
def adapter():
  """Provides a mock adapter for testing."""
  return NumpyAdapter()


def test_numpy_examples_structure(adapter):
  """Verifies the behavior of NumPy examples structure."""
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter):
  """Verifies the behavior of tier1 math validity."""
  code = adapter.get_tiered_examples()["tier1_math"]
  ast.parse(code)
  assert "import numpy as np" in code
  assert "np.matmul" in code
  assert "np.abs" in code
  assert "np.linalg.norm" in code


def test_tier2_neural_validity(adapter):
  """Verifies the behavior of tier2 neural validity."""
  code = adapter.get_tiered_examples()["tier2_neural"]
  ast.parse(code)
  assert "Out of Scope" in code
  assert "NumPy" in code


def test_tier3_extras_validity(adapter):
  """Verifies the behavior of tier3 extras validity."""
  code = adapter.get_tiered_examples()["tier3_extras"]
  ast.parse(code)
  assert "np.save" in code
  assert "np.load" in code
  assert "file=" in code
