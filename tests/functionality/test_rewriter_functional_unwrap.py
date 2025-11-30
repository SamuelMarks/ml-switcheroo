"""
Tests for Functional Unwrapping logic in PivotRewriter.

Verifies that Flax Linen style 'functional' calls are converted to
OOP style (NNX/Torch) calls:
1. `layer.apply(vars, x)` -> `layer(x)`.
2. `y, updates = layer.apply(...)` -> `y = layer(...)`.
3. Ensures state/variables arguments are stripped.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
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
  # Target JAX (NNX) or Torch implies OOP style call
  # NOTE: source must be a valid Enum member. 'numpy' is close enough
  # to a functional source conceptually if we can't use 'flax'.
  # Actually, since Flax runs on JAX, we can conceptually call the source 'jax'.
  config = RuntimeConfig(
    source_framework="jax",  # Validated Source
    target_framework="jax",  # Targeting NNX (which is JAX but OOP)
    strict_mode=False,
  )
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter, code):
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def test_unwrap_call_only(rewriter):
  """
  Scenario: Functional call inside expression.
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
  Scenario: Tuple assignment from apply.
  Input: `y, updates = self.layer.apply(vars, x)`
  Output: `y = self.layer(x)`
  """
  code = "y, updates = self.layer.apply(vars, x)"
  result = rewrite_code(rewriter, code)

  # Must unwrap assignment target
  assert "y = self.layer(x)" in result
  assert "updates" not in result


def test_unwrap_assignment_list_style(rewriter):
  """
  Scenario: List destructuring.
  Input: `[y, state] = layer.apply(v, x)`
  Output: `y = layer(x)`
  """
  code = "[y, state] = layer.apply(v, x)"
  result = rewrite_code(rewriter, code)

  assert "y = layer(x)" in result


def test_ignore_non_apply_methods(rewriter):
  """
  Verify we don't mangle other methods.
  Input: `y = layer.forward(x)`
  Output: `y = layer.forward(x)`
  """
  code = "y = layer.forward(x)"
  result = rewrite_code(rewriter, code)

  assert "layer.forward(x)" in result


def test_standard_attributes_preserved(rewriter):
  """
  Verify attribute assignments are not touched by unwrapping logic.
  Input: `self.x, self.y = 1, 2`
  Output: `self.x, self.y = 1, 2`
  """
  code = "self.x, self.y = 1, 2"
  result = rewrite_code(rewriter, code)
  assert "self.x, self.y = 1, 2" in result


def test_simple_apply_no_args_preserved(rewriter):
  """
  Edge case: .apply() with no args.
  Input: `obj.apply()`
  Output: `obj()`
  """
  code = "obj.apply()"
  result = rewrite_code(rewriter, code)
  assert "obj()" in result
