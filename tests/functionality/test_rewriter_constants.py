"""
Tests for Attribute/Constant Rewriting.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  def __init__(self):
    # Skip init to avoid file load
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.framework_configs = {}

    # 1. Constant: float32 -> float32
    self._inject_const("float32", {"torch": "torch.float32", "jax": "jax.numpy.float32"})

    # 2. Function: abs -> abs
    self._inject_func("abs", {"torch": "torch.abs", "jax": "jax.numpy.abs"})

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

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
  return TestRewriter(MockSemantics(), config)


def rewrite(rewriter, code):
  tree = cst.parse_module(code)
  return rewriter.convert(tree).code


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


def test_function_call_rewrite(rewriter):
  """
  Input:  y = torch.abs(x)
  Expect: y = jax.numpy.abs(x)
  """
  code = "y = torch.abs(x)"
  res = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in res
  assert "torch.abs" not in res
