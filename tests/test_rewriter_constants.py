"""
Tests for Attribute/Constant Rewriting.

Verifies that:
1. Constants (torch.float32) are rewritten.
2. Functions (torch.abs) are NOT rewritten by leave_Attribute (handled by leave_Call).
3. Constants inside calls (dtype=torch.float32) are rewritten correctly.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  def __init__(self):
    # Skip init to avoid file load
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}

    # 1. Constant: float32 -> float32
    self._inject_const("float32", {"torch": "torch.float32", "jax": "jax.numpy.float32"})

    # 2. Function: abs -> abs
    self._inject_func("abs", {"torch": "torch.abs", "jax": "jax.numpy.abs"})

    # 3. Property: device -> device (mapped to string maybe?)
    # Let's map torch.device to jax.devices() function call?
    # For this test, just map to another constant pattern for simplicity.
    self._inject_const("cpu", {"torch": "torch.cpu", "jax": "jax.devices('cpu')[0]"})

  def _inject_const(self, name, mapping):
    # Constants have no std_args
    self.data[name] = {"variants": {}}
    for fw, api in mapping.items():
      self.data[name]["variants"][fw] = {"api": api}
      self._reverse_index[api] = (name, self.data[name])

  def _inject_func(self, name, mapping):
    # Functions have std_args
    self.data[name] = {"variants": {}, "std_args": ["x"]}
    for fw, api in mapping.items():
      self.data[name]["variants"][fw] = {"api": api}
      self._reverse_index[api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(MockSemantics(), config)


def rewrite(rewriter, code):
  tree = cst.parse_module(code)
  return tree.visit(rewriter).code


def test_constant_rewrite_assignment(rewriter):
  """
  Input:  dtype = torch.float32
  Expect: dtype = jax.numpy.float32
  """
  code = "x = torch.float32"
  res = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res
  assert "torch.float32" not in res


def test_constant_rewrite_argument(rewriter):
  """
  Input:  init(dtype=torch.float32)
  Expect: init(dtype=jax.numpy.float32)
  """
  code = "y = init(dtype=torch.float32)"
  res = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res


def test_function_attribute_bypass(rewriter):
  """
  Input:  f = torch.abs
  Expect: f = torch.abs (No rewrite)

  Why: By design, we skip rewriting attributes that look like functions
  (have std_args) in leave_Attribute, to avoid conflict with leave_Call.
  If 'torch.abs' is passed as an object, it's safer to leave it or handle via special rule.
  Currently, we prevent leave_Attribute from messing up leave_Call.
  """
  code = "f = torch.abs"
  res = rewrite(rewriter, code)
  # Should remain unchanged because it has std_args
  assert "torch.abs" in res


def test_function_call_rewrite(rewriter):
  """
  Input:  y = torch.abs(x)
  Expect: y = jax.numpy.abs(x)

  Verifies that leave_Attribute skipping 'torch.abs' allowed leave_Call to handle it.
  """
  code = "y = torch.abs(x)"
  res = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in res
  assert "torch.abs" not in res
