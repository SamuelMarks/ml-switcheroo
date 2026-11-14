"""Test suite for rewriter calls transformers."""

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
  """Verify `Call` handles apply index select as expected."""
  node = cst.Call(func=cst.Name("foo"), args=[])
  result = apply_index_select(node, 1)

  assert isinstance(result, cst.Subscript)
  assert result.value == node
  slice_node = result.slice[0].slice
  assert isinstance(slice_node, cst.Index)
  assert isinstance(slice_node.value, cst.Integer)
  assert slice_node.value.value == "1"


def test_rewrite_as_inline_lambda_success() -> None:
  """Verify `rewrite_as_inline_lambda` handles success as expected."""
  args = [cst.Arg(value=cst.Name("x"))]
  result = rewrite_as_inline_lambda("lambda a: a + 1", args)

  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Lambda)
  assert result.args == args


def test_rewrite_as_inline_lambda_syntax_error() -> None:
  """Verify `rewrite_as_inline_lambda` raises `ValueError` during syntax error."""
  args: list[cst.Arg] = []
  with pytest.raises(ValueError, match="Invalid lambda syntax"):
    rewrite_as_inline_lambda("lambda x y:", args)


def test_macro_substitution_transformer() -> None:
  """Verify `MacroSubstitutionTransformer` handles macro substitution transformer as expected."""
  arg_map: dict[str, cst.BaseExpression] = {"x": cst.Integer("42")}
  transformer = MacroSubstitutionTransformer(arg_map)

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
  """Verify `rewrite_as_macro` handles success as expected."""
  template = "{x} * jax.nn.sigmoid({x})"
  args = [cst.Arg(value=cst.Name("my_var"))]
  std_arg_names = ["x"]

  result = rewrite_as_macro(template, args, std_arg_names)
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.left, cst.Name)
  assert result.left.value == "my_var"
  assert isinstance(result.right, cst.Call)


def test_rewrite_as_macro_missing_arg() -> None:
  """Verify `Arg` raises `ValueError` during rewrite as macro missing arg."""
  template = "{y} + 1"
  args = [cst.Arg(value=cst.Name("x"))]
  std_arg_names = ["x"]

  with pytest.raises(ValueError, match="requires argument 'y'"):
    rewrite_as_macro(template, args, std_arg_names)


def test_rewrite_as_macro_invalid_syntax() -> None:
  """Verify `Arg` raises `ValueError` during rewrite as macro invalid syntax."""
  template = "{x} + + -"
  args = [cst.Arg(value=cst.Name("my_var"))]
  std_arg_names = ["x"]

  with pytest.raises(ValueError, match="Macro template output produced invalid python"):
    rewrite_as_macro(template, args, std_arg_names)


def test_rewrite_as_infix_unary() -> None:
  """Verify `Call` handles rewrite as infix unary as expected."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x"))]

  result = rewrite_as_infix(original_node, args, "-", ["a"])
  assert isinstance(result, cst.UnaryOperation)
  assert isinstance(result.operator, cst.Minus)
  assert cst.ensure_type(result.expression, cst.Name).value == "x"


def test_rewrite_as_infix_unary_wrapped() -> None:
  """Verify `Call` handles rewrite as infix unary wrapped as expected."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  bin_op = cst.BinaryOperation(left=cst.Name("y"), operator=cst.Add(), right=cst.Name("z"))
  args = [cst.Arg(value=bin_op)]

  result = rewrite_as_infix(original_node, args, "-", ["a"])
  assert isinstance(result, cst.UnaryOperation)
  assert isinstance(result.expression, cst.BinaryOperation)
  assert len(result.expression.lpar) > 0


def test_rewrite_as_infix_unary_missing_args() -> None:
  """Verify `Call` raises `ValueError` during rewrite as infix unary missing args."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  with pytest.raises(ValueError, match="Unary operator '-' expects 1 argument"):
    rewrite_as_infix(original_node, [], "-", ["a"])


def test_rewrite_as_infix_unary_unsupported() -> None:
  """Verify `Call` raises `ValueError` during rewrite as infix unary unsupported."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x"))]
  with pytest.raises(ValueError, match="Unsupported unary operator: x"):
    rewrite_as_infix(original_node, args, "x", ["a"])


def test_rewrite_as_infix_binary() -> None:
  """Verify `Call` handles rewrite as infix binary as expected."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]

  result = rewrite_as_infix(original_node, args, "+", ["a", "b"])
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.operator, cst.Add)
  assert cst.ensure_type(result.left, cst.Name).value == "x"
  assert cst.ensure_type(result.right, cst.Name).value == "y"


def test_rewrite_as_infix_binary_missing_args() -> None:
  """Verify `Call` raises `ValueError` during rewrite as infix binary missing args."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x"))]
  with pytest.raises(ValueError, match="Binary operator '\\+' requires 2 arguments"):
    rewrite_as_infix(original_node, args, "+", ["a", "b"])


def test_rewrite_as_infix_binary_unsupported() -> None:
  """Verify `Call` raises `ValueError` during rewrite as infix binary unsupported."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]
  with pytest.raises(ValueError, match="Unsupported binary operator: x"):
    rewrite_as_infix(original_node, args, "x", ["a", "b"])


def test_rewrite_as_infix_wrong_arity() -> None:
  """Verify `Call` raises `ValueError` during rewrite as infix wrong arity."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y")), cst.Arg(value=cst.Name("z"))]
  with pytest.raises(ValueError, match="Infix operator requires 1 or 2 args"):
    rewrite_as_infix(original_node, args, "+", ["a", "b", "c"])


def test_rewrite_as_infix_no_std_args() -> None:
  """Verify `Call` handles rewrite as infix no std args as expected."""
  original_node = cst.Call(func=cst.Name("foo"), args=[])
  args = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]

  result = rewrite_as_infix(original_node, args, "+", [])
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.operator, cst.Add)
  assert cst.ensure_type(result.left, cst.Name).value == "x"
  assert cst.ensure_type(result.right, cst.Name).value == "y"
  assert cst.Module([]).code_for_node(result) == "x + y"


def test_rewrite_as_macro_non_expression_result_raises(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `rewrite_as_macro` raises `MacroExpansionError` if substitution returns a non-`BaseExpression`.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  from ml_switcheroo.core.rewriter.calls.transformers import MacroSubstitutionTransformer
  from ml_switcheroo.exceptions import MacroExpansionError

  monkeypatch.setattr(MacroSubstitutionTransformer, "leave_Name", lambda self, o, u: cst.RemoveFromParent())
  with pytest.raises(MacroExpansionError, match="expected BaseExpression"):
    rewrite_as_macro("{x}", [cst.Arg(value=cst.Name("a"))], ["x"])
