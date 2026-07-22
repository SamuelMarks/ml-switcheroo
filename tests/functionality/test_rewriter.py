"""Test suite for the Rewriter module."""

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
    self.import_data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.framework_configs = {}
    self._inject("abs", ["x"], "torch.abs", "jax.numpy.abs")
    self._inject("sum", ["x"], "torch.sum", "jax.numpy.sum", s_args={"x": "input"}, t_args={"x": "a"})
    self._inject("neg", ["x"], "torch.neg", "jax.numpy.negative")
    self._inject("add", ["x", "y"], "torch.add", "jax.numpy.add")

  def get_framework_config(self, framework: str):
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject(self, name, std_args, s_api, t_api, s_args=None, t_args=None):
    """Mock implementation of  inject."""
    s_def = {"api": s_api}
    if s_args:
      s_def["args"] = s_args
    t_def = {"api": t_api}
    if t_args:
      t_def["args"] = t_args
    self.data[name] = {"std_args": std_args, "variants": {"torch": s_def, "jax": t_def}}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  return TestRewriter(semantics, config)


def rewrite(rewriter, code):
  """Rewrites ."""
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return new_tree.code


def test_simple_api_swap(rewriter):
  """Verifies the behavior of simple API swap."""
  code = "y = torch.abs(x)"
  result = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_argument_renaming(rewriter):
  """Verifies the behavior of argument renaming."""
  code = "y = torch.sum(input=t)"
  result = rewrite(rewriter, code)
  assert "jax.numpy.sum(a=t)" in result


def test_nested_calls_recursive(rewriter):
  """Verifies the behavior of nested calls recursive."""
  code = "y = torch.abs(torch.neg(x))"
  result = rewrite(rewriter, code)
  assert "jax.numpy.abs" in result
  assert "jax.numpy.negative(x)" in result
  assert "torch" not in result


def test_complex_nested_structure(rewriter):
  """Verifies the behavior of complex nested structure."""
  code = "y = torch.add(torch.abs(a), torch.neg(b))"
  result = rewrite(rewriter, code)
  assert "jax.numpy.add" in result
  assert "jax.numpy.abs(a)" in result
  assert "jax.numpy.negative(b)" in result


def test_return_statement_rewrite(rewriter):
  """Verifies the behavior of return statement rewrite."""
  code = "def f(x):\n    return torch.abs(x)"
  result = rewrite(rewriter, code)
  assert "return jax.numpy.abs(x)" in result


def test_function_arg_rewrite(rewriter):
  """Verifies the behavior of function argument rewrite."""
  code = "print(torch.abs(x))"
  result = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_list_element_rewrite(rewriter):
  """Verifies the behavior of list element rewrite."""
  code = "l = [torch.abs(x), torch.neg(y)]"
  result = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result
  assert "jax.numpy.negative(y)" in result


def test_dict_value_rewrite(rewriter):
  """Verifies the behavior of dictionary value rewrite."""
  code = "d = {'val': torch.abs(x)}"
  result = rewrite(rewriter, code)
  assert "{'val': jax.numpy.abs(x)}" in result


def test_pass_through_unknown(rewriter):
  """Verifies the behavior of pass through unknown."""
  code = "y = torch.unknown_func(x)"
  result = rewrite(rewriter, code)
  assert "torch.unknown_func(x)" in result


def test_aliased_usage(rewriter):
  """Verifies the behavior of aliased usage."""
  code = "\nimport torch as t\ny = t.abs(x)\n"
  result = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result
