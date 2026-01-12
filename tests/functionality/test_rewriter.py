"""
Comprehensive Integration Tests for the PivotRewriter (TestRewriter).

Verifies:
1.  **API Swapping**: Calls are correctly mapped (torch.abs -> jax.numpy.abs).
2.  **Arg Normalization**: Params are pivoted via the abstract spec (input -> x -> a).
3.  **Recursive Rewriting**: Nested calls (`abs(neg(x))`) are transformed inside-out.
4.  **Complex Statements**: Calls inside Return, Assign, and List structures work.
5.  **Alias Resolution**: Integration with alias map works end-to-end.
6.  **Pass-through**: Unknown APIs are preserved.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  """
  Mock Manager that skips file I/O and provides deterministic test data.
  """

  def __init__(self):
    # Skip super() init to avoid loading real files
    self.data = {}
    self.import_data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.framework_configs = {}

    # 1. Simple Swap: abs
    # torch.abs(x) -> jax.numpy.abs(x)
    self._inject("abs", ["x"], "torch.abs", "jax.numpy.abs")

    # 2. Argument Rename: sum
    # Torch: sum(input) -> Std: sum(x) -> Jax: sum(a)
    self._inject("sum", ["x"], "torch.sum", "jax.numpy.sum", s_args={"x": "input"}, t_args={"x": "a"})

    # 3. Unary Op: neg
    self._inject("neg", ["x"], "torch.neg", "jax.numpy.negative")

    # 4. Binary Op: add
    self._inject("add", ["x", "y"], "torch.add", "jax.numpy.add")

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _inject(self, name, std_args, s_api, t_api, s_args=None, t_args=None):
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
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  return PivotRewriter(semantics, config)


def rewrite(rewriter, code):
  """Helper to parse and return code string."""
  tree = cst.parse_module(code)
  # Using pipeline conversion via shim
  new_tree = rewriter.convert(tree)
  return new_tree.code


def test_simple_api_swap(rewriter):
  """
  Input:  y = torch.abs(x)
  Output: y = jax.numpy.abs(x)
  """
  code = "y = torch.abs(x)"
  result = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_argument_renaming(rewriter):
  """
  Input:  y = torch.sum(input=t)
  Output: y = jax.numpy.sum(a=t)
  """
  code = "y = torch.sum(input=t)"
  result = rewrite(rewriter, code)
  assert "jax.numpy.sum(a=t)" in result


def test_nested_calls_recursive(rewriter):
  """
  Input:  y = torch.abs(torch.neg(x))
  Output: y = jax.numpy.abs(jax.numpy.negative(x))
  """
  code = "y = torch.abs(torch.neg(x))"
  result = rewrite(rewriter, code)

  assert "jax.numpy.abs" in result
  assert "jax.numpy.negative(x)" in result
  assert "torch" not in result


def test_complex_nested_structure(rewriter):
  """
  Input:  y = torch.add(torch.abs(a), torch.neg(b))
  Output is fully converted.
  """
  code = "y = torch.add(torch.abs(a), torch.neg(b))"
  result = rewrite(rewriter, code)

  assert "jax.numpy.add" in result
  assert "jax.numpy.abs(a)" in result
  assert "jax.numpy.negative(b)" in result


def test_return_statement_rewrite(rewriter):
  """
  Verify rewrites work inside return statements.
  Input:  return torch.abs(x)
  Output: return jax.numpy.abs(x)
  """
  code = "def f(x):\n    return torch.abs(x)"
  result = rewrite(rewriter, code)
  assert "return jax.numpy.abs(x)" in result


def test_function_arg_rewrite(rewriter):
  """
  Verify rewrites work when call is an argument to another function.
  Input:  print(torch.abs(x))
  Output: print(jax.numpy.abs(x))
  """
  code = "print(torch.abs(x))"
  result = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_list_element_rewrite(rewriter):
  """
  Input:  l = [torch.abs(x), torch.neg(y)]
  Output: l = [jax.numpy.abs(x), jax.numpy.negative(y)]
  """
  code = "l = [torch.abs(x), torch.neg(y)]"
  result = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result
  assert "jax.numpy.negative(y)" in result


def test_dict_value_rewrite(rewriter):
  """
  Input:  d = {'val': torch.abs(x)}
  Output: d = {'val': jax.numpy.abs(x)}
  """
  code = "d = {'val': torch.abs(x)}"
  result = rewrite(rewriter, code)
  assert "{'val': jax.numpy.abs(x)}" in result


def test_pass_through_unknown(rewriter):
  """
  Verify unknown APIs are preserved verbatim.
  """
  code = "y = torch.unknown_func(x)"
  result = rewrite(rewriter, code)
  assert "torch.unknown_func(x)" in result


def test_aliased_usage(rewriter):
  """
  Verify that local aliases defined in import override source rules.
  Input:
      import torch as t
      y = t.abs(x)
  Output:
      ...
      y = jax.numpy.abs(x)
  """
  code = """
import torch as t
y = t.abs(x)
"""
  result = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result
