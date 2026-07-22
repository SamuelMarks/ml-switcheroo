"""Test suite for the Rewriter Constants module."""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.framework_configs = {}
    self._inject_const("float32", {"torch": "torch.float32", "jax": "jax.numpy.float32"})
    self._inject_func("abs", {"torch": "torch.abs", "jax": "jax.numpy.abs"})

  def get_framework_config(self, framework: str):
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject_const(self, name, mapping):
    """Mock implementation of  inject const."""
    self.data[name] = {"variants": {}}
    for fw, api in mapping.items():
      self.data[name]["variants"][fw] = {"api": api}
      self._reverse_index[api] = (name, self.data[name])

  def _inject_func(self, name, mapping):
    """Mock implementation of  inject function."""
    self.data[name] = {"variants": {}, "std_args": ["x"]}
    for fw, api in mapping.items():
      self.data[name]["variants"][fw] = {"api": api}
      self._reverse_index[api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(MockSemantics(), config)


def rewrite(rewriter, code):
  """Rewrites ."""
  tree = cst.parse_module(code)
  return rewriter.convert(tree).code


def test_constant_rewrite_assignment(rewriter):
  """Verifies the behavior of constant rewrite assignment."""
  code = "x = torch.float32"
  res = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res
  assert "torch.float32" not in res


def test_constant_rewrite_argument(rewriter):
  """Verifies the behavior of constant rewrite argument."""
  code = "y = init(dtype=torch.float32)"
  res = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res


def test_function_call_rewrite(rewriter):
  """Verifies the behavior of function call rewrite."""
  code = "y = torch.abs(x)"
  res = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in res
  assert "torch.abs" not in res
