"""
Tests for Functional Unwrapping logic in TestRewriter.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockUnwrapSemantics(SemanticsManager):
  """
  Mock Manager for unwrapping tests.
  """

  def __init__(self):
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}  # No special config implies OOP default


@pytest.fixture
def rewriter():
  semantics = MockUnwrapSemantics()
  config = RuntimeConfig(
    source_framework="jax",  # Validated Source
    target_framework="jax",  # Targeting NNX (which is JAX but OOP)
    strict_mode=False,
  )
  return TestRewriter(semantics, config)


def rewrite_code(rewriter, code):
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def test_unwrap_call_only(rewriter):
  """
  Input: `z = self.layer.apply(variables, x) + 1`
  Output: `z = self.layer(x) + 1`
  """
  code = "z = self.layer.apply(variables, x) + 1"
  result = rewrite_code(rewriter, code)

  assert "self.layer(x)" in result
  assert "apply" not in result
  assert "variables" not in result


def test_unwrap_assignment_tuple(rewriter):
  """
  Input: `y, updates = self.layer.apply(vars, x)`
  Output: `y = self.layer(x)`
  """
  code = "y, updates = self.layer.apply(vars, x)"
  result = rewrite_code(rewriter, code)

  # Must unwrap assignment target
  assert "y = self.layer(x)" in result
  assert "updates" not in result
