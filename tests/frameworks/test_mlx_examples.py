"""Test suite for the Mlx Examples module."""

import ast
import pytest
from ml_switcheroo.frameworks.mlx import MLXAdapter


@pytest.fixture
def adapter():
  """Provides a mock adapter for testing."""
  return MLXAdapter()


def test_mlx_examples_structure(adapter):
  """Verifies the behavior of MLX examples structure."""
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter):
  """Verifies the behavior of tier1 math validity."""
  code = adapter.get_tiered_examples()["tier1_math"]
  ast.parse(code)
  assert "import mlx.core as mx" in code
  assert "mx.abs" in code
  assert "mx.add" in code
  assert "mx.mean" in code


def test_tier2_neural_validity(adapter):
  """Verifies the behavior of tier2 neural validity."""
  code = adapter.get_tiered_examples()["tier2_neural"]
  ast.parse(code)
  assert "class MLP(nn.Module):" in code
  assert "def __init__(self" in code
  assert "super().__init__()" in code
  assert "def __call__(self, x):" in code
  assert "nn.Linear" in code


def test_tier3_extras_validity(adapter):
  """Verifies the behavior of tier3 extras validity."""
  code = adapter.get_tiered_examples()["tier3_extras"]
  ast.parse(code)
  assert "with mx.stream(mx.gpu):" in code
  assert "mx.eval(y)" in code
