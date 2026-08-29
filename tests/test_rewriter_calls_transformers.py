"""Test module."""

from typing import Dict, List

import libcst as cst
import pytest

from ml_switcheroo.core.rewriter.calls.transformers import (
  MacroSubstitutionTransformer,
  apply_index_select,
  rewrite_as_infix,
  rewrite_as_inline_lambda,
  rewrite_as_macro,
)


def test_apply_index_select() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  result: cst.Subscript = apply_index_select(node, 1)

  assert isinstance(result, cst.Subscript)
  assert result.value == node
  assert isinstance(result.slice[0].slice.value, cst.Integer)
  assert result.slice[0].slice.value.value == "1"


def test_rewrite_as_inline_lambda_success() -> None:
  """Docstring."""
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("x"))]
  result: cst.Call = rewrite_as_inline_lambda("lambda a: a + 1", args)

  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Lambda)
  assert result.args == args


def test_rewrite_as_inline_lambda_syntax_error() -> None:
  """Docstring."""
  args: List[cst.Arg] = []
  with pytest.raises(ValueError, match="Invalid lambda syntax"):
    rewrite_as_inline_lambda("lambda x y:", args)


def test_macro_substitution_transformer() -> None:
  """Docstring."""
  arg_map: Dict[str, cst.BaseExpression] = {"x": cst.Integer("42")}
  transformer: MacroSubstitutionTransformer = MacroSubstitutionTransformer(arg_map)

  # Matching name
  original_node: cst.Name = cst.Name("_MACRO_VAR_x_")
  updated_node: cst.Name = cst.Name("_MACRO_VAR_x_")
  result: cst.BaseExpression = transformer.leave_Name(original_node, updated_node)
  assert result == arg_map["x"]

  # Non-matching name
  original_node2: cst.Name = cst.Name("_MACRO_VAR_y_")
  updated_node2: cst.Name = cst.Name("_MACRO_VAR_y_")
  result2: cst.BaseExpression = transformer.leave_Name(original_node2, updated_node2)
  assert result2 == updated_node2

  # Not a macro var
  original_node3: cst.Name = cst.Name("x")
  updated_node3: cst.Name = cst.Name("x")
  result3: cst.BaseExpression = transformer.leave_Name(original_node3, updated_node3)
  assert result3 == updated_node3


def test_rewrite_as_macro_success() -> None:
  """Docstring."""
  template: str = "{x} * jax.nn.sigmoid({x})"
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("my_var"))]
  std_arg_names: List[str] = ["x"]

  result: cst.BaseExpression = rewrite_as_macro(template, args, std_arg_names)
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.left, cst.Name)
  assert result.left.value == "my_var"
  assert isinstance(result.right, cst.Call)


def test_rewrite_as_macro_missing_arg() -> None:
  """Docstring."""
  template: str = "{y} + 1"
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("x"))]
  std_arg_names: List[str] = ["x"]

  with pytest.raises(ValueError, match="requires argument 'y'"):
    rewrite_as_macro(template, args, std_arg_names)


def test_rewrite_as_macro_invalid_syntax() -> None:
  """Docstring."""
  template: str = "{x} + + -"
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("my_var"))]
  std_arg_names: List[str] = ["x"]

  with pytest.raises(ValueError, match="Macro template output produced invalid python"):
    rewrite_as_macro(template, args, std_arg_names)


def test_rewrite_as_infix_unary() -> None:
  """Docstring."""
  original_node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("x"))]

  result: cst.BaseExpression = rewrite_as_infix(original_node, args, "-", ["a"])
  assert isinstance(result, cst.UnaryOperation)
  assert isinstance(result.operator, cst.Minus)
  assert getattr(result.expression, "value", None) == "x"


def test_rewrite_as_infix_unary_wrapped() -> None:
  """Docstring."""
  original_node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  bin_op: cst.BinaryOperation = cst.BinaryOperation(left=cst.Name("y"), operator=cst.Add(), right=cst.Name("z"))
  args: List[cst.Arg] = [cst.Arg(value=bin_op)]

  result: cst.BaseExpression = rewrite_as_infix(original_node, args, "-", ["a"])
  assert isinstance(result, cst.UnaryOperation)
  assert isinstance(result.expression, cst.BinaryOperation)
  assert len(result.expression.lpar) > 0


def test_rewrite_as_infix_unary_missing_args() -> None:
  """Docstring."""
  original_node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  with pytest.raises(ValueError, match="Unary operator '-' expects 1 argument"):
    rewrite_as_infix(original_node, [], "-", ["a"])


def test_rewrite_as_infix_unary_unsupported() -> None:
  """Docstring."""
  original_node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("x"))]
  with pytest.raises(ValueError, match="Unsupported unary operator: x"):
    rewrite_as_infix(original_node, args, "x", ["a"])


def test_rewrite_as_infix_binary() -> None:
  """Docstring."""
  original_node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]

  result: cst.BaseExpression = rewrite_as_infix(original_node, args, "+", ["a", "b"])
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.operator, cst.Add)
  assert getattr(result.left, "value", None) == "x"
  assert getattr(result.right, "value", None) == "y"


def test_rewrite_as_infix_binary_missing_args() -> None:
  """Docstring."""
  original_node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("x"))]
  with pytest.raises(ValueError, match="Binary operator '\\+' requires 2 arguments"):
    rewrite_as_infix(original_node, args, "+", ["a", "b"])


def test_rewrite_as_infix_binary_unsupported() -> None:
  """Docstring."""
  original_node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]
  with pytest.raises(ValueError, match="Unsupported binary operator: x"):
    rewrite_as_infix(original_node, args, "x", ["a", "b"])


def test_rewrite_as_infix_wrong_arity() -> None:
  """Docstring."""
  original_node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y")), cst.Arg(value=cst.Name("z"))]
  with pytest.raises(ValueError, match="Infix operator requires 1 or 2 args"):
    rewrite_as_infix(original_node, args, "+", ["a", "b", "c"])


def test_rewrite_as_infix_no_std_args() -> None:
  """Docstring."""
  original_node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]

  result: cst.BaseExpression = rewrite_as_infix(original_node, args, "+", [])
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.operator, cst.Add)
