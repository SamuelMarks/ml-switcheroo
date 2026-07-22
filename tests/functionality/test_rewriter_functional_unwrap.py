"""Test suite for the Rewriter Functional Unwrap module."""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockUnwrapSemantics(SemanticsManager):
  """Mock Unwrap Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockUnwrapSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  semantics = MockUnwrapSemantics()
  config = RuntimeConfig(source_framework="jax", target_framework="jax", strict_mode=False)
  return TestRewriter(semantics, config)


def rewrite_code(rewriter, code):
  """Rewrites code."""
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def test_unwrap_call_only(rewriter):
  """Verifies the behavior of unwrap call only."""
  code = "z = self.layer.apply(variables, x) + 1"
  result = rewrite_code(rewriter, code)
  assert "self.layer(x)" in result
  assert "apply" not in result
  assert "variables" not in result


def test_unwrap_assignment_tuple(rewriter):
  """Verifies the behavior of unwrap assignment tuple."""
  code = "y, updates = self.layer.apply(vars, x)"
  result = rewrite_code(rewriter, code)
  assert "y = self.layer(x)" in result
  assert "updates" not in result
