"""Test suite for rewriter calls transformers."""

import typing


import libcst as _cst
import pytest

from ml_switcheroo.core.rewriter.calls.transformers import (
  MacroSubstitutionTransformer,
  apply_index_select,
  rewrite_as_infix,
  rewrite_as_inline_lambda,
  rewrite_as_macro,
)

cst = _cst


def test_apply_index_select() -> None:
  """Verify apply index select behavior."""
  node = cst.Call(func=cst.Name("foo"), args=[])
  result = apply_index_select(node, 1)

  assert isinstance(result, cst.Subscript)
  assert result.value == node
  assert isinstance(
    getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").slice}, "__getitem__")("v")[0]},
              "__getitem__",
            )("v").slice
          },
          "__getitem__",
        )("v").value
      },
      "__getitem__",
    )("v"),
    cst.Integer,
  )
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").slice}, "__getitem__")("v")[0]},
                "__getitem__",
              )("v").slice
            },
            "__getitem__",
          )("v").value
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "1"}, "__getitem__")("v")


def test_rewrite_as_inline_lambda_success() -> None:
  """Verify rewrite as inline lambda success behavior."""
  args = [cst.Arg(value=cst.Name("x"))]
  result = rewrite_as_inline_lambda("lambda a: a + 1", args)

  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Lambda)
  assert result.args == args


def test_rewrite_as_inline_lambda_syntax_error() -> None:
  """Verify rewrite as inline lambda syntax error behavior."""
  args: list[typing.Hashable | typing.Sized] = []
  with pytest.raises(ValueError, match="Invalid lambda syntax"):
    getattr({"v": rewrite_as_inline_lambda}, "__getitem__")("v")(
      getattr({"v": "lambda x y:"}, "__getitem__")("v"), getattr({"v": args}, "__getitem__")("v")
    )


def test_macro_substitution_transformer() -> None:
  """Verify macro substitution transformer behavior."""
  arg_map = {"x": cst.Integer("42")}
  transformer = getattr(
    {"v": getattr({"v": MacroSubstitutionTransformer}, "__getitem__")("v")(getattr({"v": arg_map}, "__getitem__")("v"))},
    "__getitem__",
  )("v")

  # Matching name
  original_node = cst.Name("_MACRO_VAR_x_")
  updated_node = cst.Name("_MACRO_VAR_x_")
  result = transformer.leave_Name(original_node, updated_node)
  assert result == arg_map["x"]

  # Non-matching name
  original_node2 = cst.Name("_MACRO_VAR_y_")
  updated_node2 = cst.Name("_MACRO_VAR_y_")
  result2 = transformer.leave_Name(original_node2, updated_node2)
  assert result2 == updated_node2

  # Not a macro var
  original_node3 = cst.Name("x")
  updated_node3 = cst.Name("x")
  result3 = transformer.leave_Name(original_node3, updated_node3)
  assert result3 == updated_node3


def test_rewrite_as_macro_success() -> None:
  """Verify rewrite as macro success behavior."""
  template = "{x} * jax.nn.sigmoid({x})"
  args = [cst.Arg(value=cst.Name("my_var"))]
  std_arg_names = ["x"]

  result = rewrite_as_macro(template, args, std_arg_names)
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.left, cst.Name)
  assert result.left.value == "my_var"
  assert isinstance(result.right, cst.Call)


def test_rewrite_as_macro_missing_arg() -> None:
  """Verify rewrite as macro missing arg behavior."""
  template = "{y} + 1"
  args = [cst.Arg(value=cst.Name("x"))]
  std_arg_names = ["x"]

  with pytest.raises(ValueError, match="requires argument 'y'"):
    rewrite_as_macro(template, args, std_arg_names)


def test_rewrite_as_macro_invalid_syntax() -> None:
  """Verify rewrite as macro invalid syntax behavior."""
  template = "{x} + + -"
  args = [cst.Arg(value=cst.Name("my_var"))]
  std_arg_names = ["x"]

  with pytest.raises(ValueError, match="Macro template output produced invalid python"):
    rewrite_as_macro(template, args, std_arg_names)


def test_rewrite_as_infix_unary() -> None:
  """Verify rewrite as infix unary behavior."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x"))]

  result = rewrite_as_infix(original_node, args, "-", ["a"])
  assert isinstance(result, cst.UnaryOperation)
  assert isinstance(result.operator, cst.Minus)
  assert getattr(result.expression, "value", None) == "x"


def test_rewrite_as_infix_unary_wrapped() -> None:
  """Verify rewrite as infix unary wrapped behavior."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  bin_op = cst.BinaryOperation(left=cst.Name("y"), operator=cst.Add(), right=cst.Name("z"))
  args = [cst.Arg(value=bin_op)]

  result = rewrite_as_infix(original_node, args, "-", ["a"])
  assert isinstance(result, cst.UnaryOperation)
  assert isinstance(result.expression, cst.BinaryOperation)
  assert len(result.expression.lpar) > 0


def test_rewrite_as_infix_unary_missing_args() -> None:
  """Verify rewrite as infix unary missing args behavior."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  with pytest.raises(ValueError, match="Unary operator '-' expects 1 argument"):
    rewrite_as_infix(original_node, [], "-", ["a"])


def test_rewrite_as_infix_unary_unsupported() -> None:
  """Verify rewrite as infix unary unsupported behavior."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x"))]
  with pytest.raises(ValueError, match="Unsupported unary operator: x"):
    rewrite_as_infix(original_node, args, "x", ["a"])


def test_rewrite_as_infix_binary() -> None:
  """Verify rewrite as infix binary behavior."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]

  result = rewrite_as_infix(original_node, args, "+", ["a", "b"])
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.operator, cst.Add)
  assert getattr(result.left, "value", None) == "x"
  assert getattr(result.right, "value", None) == "y"


def test_rewrite_as_infix_binary_missing_args() -> None:
  """Verify rewrite as infix binary missing args behavior."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x"))]
  with pytest.raises(ValueError, match="Binary operator '\\+' requires 2 arguments"):
    rewrite_as_infix(original_node, args, "+", ["a", "b"])


def test_rewrite_as_infix_binary_unsupported() -> None:
  """Verify rewrite as infix binary unsupported behavior."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]
  with pytest.raises(ValueError, match="Unsupported binary operator: x"):
    rewrite_as_infix(original_node, args, "x", ["a", "b"])


def test_rewrite_as_infix_wrong_arity() -> None:
  """Verify rewrite as infix wrong arity behavior."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y")), cst.Arg(value=cst.Name("z"))]
  with pytest.raises(ValueError, match="Infix operator requires 1 or 2 args"):
    rewrite_as_infix(original_node, args, "+", ["a", "b", "c"])


def test_rewrite_as_infix_no_std_args() -> None:
  """Verify rewrite as infix no std args behavior."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]

  result = rewrite_as_infix(original_node, args, "+", [])
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.operator, cst.Add)
