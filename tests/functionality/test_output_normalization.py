"""
Tests for Output Normalization Logic (The 'Output Adapter' Pivot).

This module validates the mechanism for reconciling return signature differences
between frameworks. For example, if Framework A returns `(values, indices)` and
Framework B returns only `values`, the rewriter must wrap the call in an adapter.

The `output_adapter` field in the Semantics Schema allows defining a Python
lambda string (e.g., `lambda x: x[0]`) which the `PivotRewriter` applies
to the transformed call node.

Coverage:
1.  **Tuple Selection**: Wrapping a call to select a specific tuple index.
2.  **Tuple Reordering**: Swapping elements in a return tuple.
3.  **Dictionary Access**: Adapting dict returns to tuple returns or values.
4.  **Integration**: Ensuring adapters work with argument renaming and infix ops.
5.  **Robustness**: Handling syntax errors in adapter strings.
6.  **Composition**: Wrappers applied correctly in nested expressions.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockOutputSemantics(SemanticsManager):
  """
  Mock Manager injected with specific output adapter scenarios.
  Bypasses file loading to provide deterministic test data.
  """

  def __init__(self):
    # Skip super().__init__ to avoid loading real JSON files
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}

    # Scenario 1: 'max'
    # Torch `max` returns (values, indices) named tuple.
    # JAX `max` returns only values.
    # We simulate a target that returns a tuple, but we only want index 0.
    # Adapter: lambda x: x[0]
    self._inject("max", "torch.max", "jax.numpy.max", adapter="lambda x: x[0]")

    # Scenario 2: 'sort'
    # We pretend target returns (indices, values) but we want (values, indices).
    # Adapter: lambda x: (x[1], x[0])
    self._inject("sort", "torch.sort", "jax.numpy.sort", adapter="lambda x: (x[1], x[0])")

    # Scenario 3: 'min_idx'
    # Target returns dict {'val': v, 'idx': i}, we want just 'idx'.
    # Adapter: lambda x: x['idx']
    self._inject("min_indices", "torch.argmin", "jax.numpy.argmin_structure", adapter="lambda x: x['idx']")

    # Scenario 4: 'bad_syntax'
    # Adapter string is malformed to test error handling.
    self._inject("bad", "torch.bad", "jax.bad", adapter="lambda x: x[???")

    # Scenario 5: Infix Operator + Output Adapter
    # Example: `dot` transformation to `@` operator, then wrapped.
    # (Rare, but architecturally possible).
    self._inject(
      "dot",
      "torch.dot",
      "original.dot",  # Not used in infix transform usually but holder needed
      adapter="lambda x: float(x)",
      transform_type="infix",
      operator="@",
    )

    # Scenario 6: Structured Field (output_select_index)
    # Using the new safe DSL field for tuple destructuring
    self._inject(
      "split_vals",
      "torch.split",
      "jax.numpy.split",
      adapter=None,  # no string adapter
      select_index=0,
    )

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _inject(self, name, s_api, t_api, adapter=None, transform_type=None, operator=None, select_index=None):
    variants = {"torch": {"api": s_api}, "jax": {"api": t_api}}

    target_var = variants["jax"]
    if adapter:
      target_var["output_adapter"] = adapter
    if transform_type:
      target_var["transformation_type"] = transform_type
    if operator:
      target_var["operator"] = operator
    if select_index is not None:
      target_var["output_select_index"] = select_index

    self.data[name] = {"variants": variants, "std_args": ["x", "y"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  """Creates a pre-configured PivotRewriter for testing."""
  semantics = MockOutputSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return PivotRewriter(semantics, config)


def rewrite(rewriter, code):
  """Helper to run the rewriter on a code snippet."""
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_simple_index_wrapping(rewriter):
  """
  Scenario: `val = torch.max(x)`
  Semantics: `jax.numpy.max` requires adapter `lambda x: x[0]`.
  Expectation: `val = (lambda x: x[0])(jax.numpy.max(x))`
  """
  code = "val = torch.max(input_tensor)"
  result = rewrite(rewriter, code)

  # 1. Check Function Renaming
  assert "jax.numpy.max(input_tensor)" in result, "Target API not found"

  # 2. Check Wrapper Lambda
  assert "(lambda x: x[0])" in result, "Lambda definition missing"

  # 3. Check Application Syntax (parentheses surrounding lambda and call)
  # Logic: (lambda ...)(call(...))
  clean_res = result.replace(" ", "")
  assert "(lambdax:x[0])(jax.numpy.max(input_tensor))" in clean_res


def test_structured_index_wrapping(rewriter):
  """
  Scenario: `res = torch.split(x)`
  Semantics: `jax.numpy.split` accessed via `output_select_index=0`.
  Expectation: `res = jax.numpy.split(x)[0]` (Clean Subscript Syntax)
  """
  code = "res = torch.split(x)"
  result = rewrite(rewriter, code)

  # Should use subscription, not lambda
  assert "jax.numpy.split(x)[0]" in result
  assert "lambda" not in result


def test_tuple_reordering_adapter(rewriter):
  """
  Scenario: `res = torch.sort(x)`
  Semantics: Adapter `lambda x: (x[1], x[0])`.
  """
  code = "res = torch.sort(x)"
  result = rewrite(rewriter, code)

  assert "jax.numpy.sort(x)" in result
  assert "(lambda x: (x[1], x[0]))" in result


def test_dictionary_access_adapter(rewriter):
  """
  Scenario: `idx = torch.argmin(x)`
  Semantics: Adapter `lambda x: x['idx']`.
  Note: Dictionary keys in string format often use different quotes.
  """
  code = "idx = torch.argmin(x)"
  result = rewrite(rewriter, code)

  assert "jax.numpy.argmin_structure(x)" in result
  assert "lambda x: x['idx']" in result


def test_invalid_adapter_syntax_handling(rewriter):
  """
  Scenario: The semantic JSON contains a syntax error in `output_adapter`.
  Expectation:
      1. The transformation fails safely (no crash).
      2. Strict Mode: The code is wrapped in an Escape Hatch.
      3. The error message explicitly mentions "Output adapter failed".
      4. Original code `torch.bad(x)` is preserved inside the hatch.
  """
  code = "y = torch.bad(x)"
  result = rewrite(rewriter, code)

  assert EscapeHatch.START_MARKER in result
  assert "Output adapter failed" in result, "Error reason missing"
  # Check original source preserved
  assert "torch.bad(x)" in result
  assert "jax.bad" not in result


def test_nested_call_with_adapter(rewriter):
  """
  Scenario: The call being wrapped is an argument to another function.
  Input: `print(torch.max(x))`
  Expectation: `print( (lambda x: x[0])(jax.numpy.max(x)) )`
  """
  code = "print(torch.max(x))"
  result = rewrite(rewriter, code)

  assert "print(" in result
  assert "(lambda x: x[0])" in result
  assert "jax.numpy.max(x)" in result


def test_chained_attribute_after_adapter(rewriter):
  """
  Scenario: Accessing an attribute on the result of a wrapped call.
  Input: `val = torch.max(x).indices`
  Logic: The wrapper returns an object (e.g. index 0 of tuple).
         The `.indices` access applies to the result of the wrapper.
  Expectation: `(lambda ...)( ... ).indices`
  """
  code = "val = torch.max(x).indices"
  result = rewrite(rewriter, code)

  # Check structure
  clean_res = result.replace(" ", "")
  expected_structure = "(lambdax:x[0])(jax.numpy.max(x)).indices"

  assert expected_structure in clean_res


def test_infix_operator_wrapping(rewriter):
  """
  Scenario: An operation rewritten as an Infix Op (`@` or `+`) ALSO has an adapter.
  Input: `z = torch.dot(a, b)`
  Semantics:
      - Transformation: `a @ b`
      - Adapter: `lambda x: float(x)`
  Expectation: `z = (lambda x: float(x))(a @ b)`

  Note: This validates that `_apply_output_adapter` handles basic Expressions,
  not just Call nodes.
  """
  code = "z = torch.dot(a, b)"
  result = rewrite(rewriter, code)

  # 1. verify infix transform happened
  assert "a @ b" in result

  # 2. verify wrapper applied
  assert "(lambda x: float(x))" in result

  # 3. verify execution order via parens
  # (lambda...)(a @ b)
  clean = result.replace(" ", "")
  assert "(lambdax:float(x))(a@b)" in clean


def test_return_statement_wrapping(rewriter):
  """
  Scenario: Returning the result of a wrapped call.
  Input: `return torch.max(x)`
  """
  code = "def f(x):\n    return torch.max(x)"
  result = rewrite(rewriter, code)

  assert "return (lambda x: x[0])(jax.numpy.max(x))" in result


def test_list_comprehension_wrapping(rewriter):
  """
  Scenario: Wrapped call inside list comprehension.
  Input: `[torch.max(i) for i in items]`
  """
  code = "l = [torch.max(i) for i in items]"
  result = rewrite(rewriter, code)

  assert "jax.numpy.max(i)" in result
  assert "(lambda x: x[0])" in result
