"""
Tests for Apple MLX Adapter Examples.

Ensures that tiered examples provided by the MLXAdapter are syntactically valid
and contain expected MLX specific patterns.
"""

import ast
import pytest
from ml_switcheroo.frameworks.mlx import MLXAdapter


@pytest.fixture
def adapter():
  return MLXAdapter()


def test_mlx_examples_structure(adapter):
  """Verify the dictionary structure of bundled examples."""
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter):
  """Verify Math example uses mlx.core."""
  code = adapter.get_tiered_examples()["tier1_math"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check imports
  assert "import mlx.core as mx" in code

  # 3. Check usage
  assert "mx.abs" in code
  assert "mx.add" in code
  assert "mx.mean" in code


def test_tier2_neural_validity(adapter):
  """Verify Neural example uses nn.Module and __call__."""
  code = adapter.get_tiered_examples()["tier2_neural"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check Class Structure
  assert "class MLP(nn.Module):" in code
  assert "def __init__(self" in code
  assert "super().__init__()" in code

  # 3. Check MLX specific method name
  assert "def __call__(self, x):" in code
  assert "nn.Linear" in code


def test_tier3_extras_validity(adapter):
  """Verify Extras example uses Stream/Device logic."""
  code = adapter.get_tiered_examples()["tier3_extras"]

  # 1. Check syntax
  ast.parse(code)

  # 2. Check Stream usage
  assert "with mx.stream(mx.gpu):" in code
  assert "mx.eval(y)" in code
