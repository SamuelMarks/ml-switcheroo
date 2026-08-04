"""Unit tests for AST transformation helper functions used in call rewriting.

This module contains pytest-based unit tests for the AST transformation
mechanisms defined in `ml_switcheroo.core.rewriter.calls.transformers`.
Specifically, it covers testing index-based slicing via `apply_index_select`,
argument substitution inside inline lambda functions via `rewrite_as_inline_lambda`,
vulnerability-safe python template evaluation via `rewrite_as_macro`, macro variable
substitution via `MacroSubstitutionTransformer`, and transforming functional call nodes
to unary and binary infix operations via `rewrite_as_infix`.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter.calls.transformers import (
  apply_index_select,
  rewrite_as_inline_lambda,
  rewrite_as_macro,
  rewrite_as_infix,
  MacroSubstitutionTransformer,
)


def test_apply_index_select():
  """Verify the correctness of `apply_index_select` AST transformation.

  This test checks that `apply_index_select` correctly wraps an expression
  node (e.g. a Call node) with a subscript expression access for a specific
  integer index. It asserts that the resulting node is a `libcst.Subscript`
  and that the inner slice is an integer match representing the given index.

  Args:
      None

  Returns:
      None
  """
  node = cst.Call(func=cst.Name("func"), args=[])
  res = apply_index_select(node, 1)
  assert isinstance(res, cst.Subscript)
  assert isinstance(res.slice[0].slice.value, cst.Integer)
  assert res.slice[0].slice.value.value == "1"


def test_rewrite_as_inline_lambda():
  """Verify Immediately Invoked Lambda Expression (IIFE) rewriting logic.

  This test ensures that `rewrite_as_inline_lambda` correctly compiles and wraps
  a target lambda string expression as a standard parenthesized lambda, and
  invokes it using the supplied CST arguments. It also asserts that calling
  the helper with syntactically malformed lambda expressions raises a `ValueError`.

  Args:
      None

  Returns:
      None
  """
  args = [cst.Arg(value=cst.Name("x"))]
  res = rewrite_as_inline_lambda("lambda x: x + 1", args)
  assert isinstance(res, cst.Call)
  with pytest.raises(ValueError, match="Invalid lambda syntax"):
    rewrite_as_inline_lambda("lambda : +++", args)


def test_rewrite_as_macro():
  """Verify structural template substitution logic of `rewrite_as_macro`.

  This test validates that macro templates (such as `"{x} + {y}"`) are
  correctly expanded using safe, structured CST substitution. It verifies
  that:
  - Proper CST nodes are successfully injected into the template's placeholders.
  - Omission of standard arguments required by the template results in a `ValueError`.
  - Syntactically malformed Python expressions inside the template prompt a `ValueError`.

  Args:
      None

  Returns:
      None
  """
  args = [cst.Arg(value=cst.Name("x_node")), cst.Arg(value=cst.Name("y_node"))]
  names = ["x", "y"]

  # Valid substitution
  res = rewrite_as_macro("{x} + {y}", args, names)
  assert isinstance(res, cst.BinaryOperation)
  assert isinstance(res.left, cst.Name)
  assert res.left.value == "x_node"

  # Missing argument in template
  with pytest.raises(ValueError, match="Macro template requires argument 'z'"):
    rewrite_as_macro("{z} + {x}", args, names)

  # Invalid python syntax inside template
  with pytest.raises(ValueError, match="invalid python"):
    rewrite_as_macro("{x} +++", args, names)


def test_macro_transformer_unmatched():
  """Verify edge cases and fallback paths in `MacroSubstitutionTransformer`.

  This test validates that Name nodes which are either not prefixed with
  `_MACRO_VAR_` or do not exist in the substitution dictionary are returned
  as-is by `MacroSubstitutionTransformer`. This guarantees correct and safe fallback
  scenarios where normal identifiers in templates are preserved untouched.

  Args:
      None

  Returns:
      None
  """
  # Test that nodes not matching safe prefix are unchanged
  transformer = MacroSubstitutionTransformer({"x": cst.Name("replacement")})
  node = cst.Name("normal_name")
  updated = transformer.leave_Name(node, node)
  assert isinstance(updated, cst.Name)
  assert updated.value == "normal_name"

  # Test matched safe prefix but not in map (should be impossible in practice because we check before)
  node2 = cst.Name("_MACRO_VAR_not_in_map_")
  updated2 = transformer.leave_Name(node2, node2)
  assert isinstance(updated2, cst.Name)
  assert updated2.value == "_MACRO_VAR_not_in_map_"


def test_rewrite_as_infix_unary():
  """Verify functional-to-unary infix operator transformation.

  This test ensures that unary operations (e.g., `-x`, `~x`) are correctly
  reconstructed from standard functional call nodes and their respective arguments.
  It asserts proper syntax validation, arity checking, handling of nested/wrapped
  binary expressions inside unary nodes, and error reporting for unsupported unary
  operators.

  Args:
      None

  Returns:
      None
  """
  original = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x"))]

  res = rewrite_as_infix(original, args, "-", ["x"])
  assert isinstance(res, cst.UnaryOperation)
  assert isinstance(res.operator, cst.Minus)

  with pytest.raises(ValueError, match="Unsupported unary operator"):
    rewrite_as_infix(original, args, "???", ["x"])

  with pytest.raises(ValueError, match="expects 1 argument"):
    rewrite_as_infix(original, [], "-", ["x"])

  # Unary wrapping a binary operation
  bin_arg = [cst.Arg(value=cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b")))]
  res2 = rewrite_as_infix(original, bin_arg, "~", ["x"])
  assert isinstance(res2.expression, cst.BinaryOperation)
  assert len(res2.expression.lpar) > 0


def test_rewrite_as_infix_binary():
  """Verify functional-to-binary infix operator transformation.

  This test ensures that binary operations (e.g., `x + y`) are correctly
  reconstructed from functional call structures using mapped infix symbols.
  It tests that:
  - Corresponding binary CST nodes (like `cst.BinaryOperation` with an `Add` operator) are built.
  - Unsupported binary operators trigger a `ValueError`.
  - Arity violations (insufficient arguments) result in `ValueError`.

  Args:
      None

  Returns:
      None
  """
  original = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]

  res = rewrite_as_infix(original, args, "+", ["x", "y"])
  assert isinstance(res, cst.BinaryOperation)
  assert isinstance(res.operator, cst.Add)

  with pytest.raises(ValueError, match="Unsupported binary operator"):
    rewrite_as_infix(original, args, "???", ["x", "y"])

  with pytest.raises(ValueError, match="requires 2 arguments"):
    rewrite_as_infix(original, [args[0]], "+", ["x", "y"])


def test_rewrite_as_infix_invalid_arity():
  """Verify error handling for invalid arities during infix rewriting.

  This test asserts that when `rewrite_as_infix` receives an invalid number of
  arguments (such as 3 arguments for an operator meant to be binary or unary),
  it correctly raises a `ValueError` indicating that infix operators only support
  1 or 2 arguments.

  Args:
      None

  Returns:
      None
  """
  original = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y")), cst.Arg(value=cst.Name("z"))]

  with pytest.raises(ValueError, match="requires 1 or 2 args"):
    rewrite_as_infix(original, args, "+", ["x", "y", "z"])
