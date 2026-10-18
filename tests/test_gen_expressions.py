"""Test suite for gen expressions."""

import typing

from unittest import mock

import libcst as _cst
import pytest

from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode, ValueNode
from ml_switcheroo.core.mlir.gen_expressions import ExpressionGeneratorMixin

cst: typing.Any = _cst


class DummyGenerator(ExpressionGeneratorMixin):
  """Test helper class DummyGenerator."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self._resolved: typing.Any = {}

  def _resolve_operand(self, ssa_name: typing.Any) -> typing.Any:
    """Execute resolve operand helper logic.

    Args:
        ssa_name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if ssa_name in self._resolved:
      return self._resolved[ssa_name]
    return cst.Name(f"resolved_{ssa_name}")


def test_resolve_operand_not_implemented() -> None:
  """Verify resolve operand not implemented behavior."""

  class IncompleteGenerator(ExpressionGeneratorMixin):
    """Test helper class IncompleteGenerator."""

    pass

  gen: typing.Any = IncompleteGenerator()
  with pytest.raises(NotImplementedError):
    gen._resolve_operand("foo")


def test_parse_keywords() -> None:
  """Verify parse keywords behavior."""
  gen: typing.Any = DummyGenerator()

  # Empty
  op: typing.Any = OperationNode(name="op", operands=[], attributes=[])
  assert gen._parse_keywords(op) == []

  # List case
  op_list: typing.Any = OperationNode(
    name="op", operands=[], attributes=[AttributeNode(name="arg_keywords", value=['"foo"', "bar"])]
  )
  assert gen._parse_keywords(op_list) == ["foo", "bar"]

  # String case
  op_str: typing.Any = OperationNode(
    name="op", operands=[], attributes=[AttributeNode(name="arg_keywords", value='["baz", "qux"]')]
  )
  assert gen._parse_keywords(op_str) == ["baz", "qux"]

  # Invalid string case
  op_inv: typing.Any = OperationNode(
    name="op", operands=[], attributes=[AttributeNode(name="arg_keywords", value="[invalid")]
  )
  assert gen._parse_keywords(op_inv) == []


def test_expr_sw_constant() -> None:
  """Verify expr sw constant behavior."""
  gen: typing.Any = DummyGenerator()

  # Valid parseable expression
  op: typing.Any = OperationNode(name="sw.constant", operands=[], attributes=[AttributeNode(name="value", value="42")])
  expr: typing.Any = gen._expr_sw_constant(op)
  assert isinstance(expr, cst.Integer)
  assert expr.value == "42"

  # Default value (0)
  op_def: typing.Any = OperationNode(name="sw.constant", operands=[], attributes=[])
  expr_def: typing.Any = gen._expr_sw_constant(op_def)
  assert isinstance(expr_def, cst.Integer)
  assert expr_def.value == "0"

  # Unparseable expression fallback to Name
  with mock.patch("libcst.parse_expression", side_effect=Exception("Failed")):
    op_inv: typing.Any = OperationNode(
      name="sw.constant", operands=[], attributes=[AttributeNode(name="value", value="fallback_name")]
    )
    expr_inv: typing.Any = gen._expr_sw_constant(op_inv)
    assert isinstance(expr_inv, cst.Name)
    assert expr_inv.value == "fallback_name"


def test_expr_sw_getattr() -> None:
  """Verify expr sw getattr behavior."""
  gen: typing.Any = DummyGenerator()

  # No operands error
  op_err: typing.Any = OperationNode(name="sw.getattr", operands=[], attributes=[])
  expr_err: typing.Any = gen._expr_sw_getattr(op_err)
  assert isinstance(expr_err, cst.Name)
  assert expr_err.value == "error"

  # Success
  op: typing.Any = OperationNode(
    name="sw.getattr", operands=[ValueNode(name="obj")], attributes=[AttributeNode(name="name", value='"my_attr"')]
  )
  expr: typing.Any = gen._expr_sw_getattr(op)
  assert isinstance(expr, cst.Attribute)
  assert isinstance(expr.value, cst.Name)
  assert expr.value.value == "resolved_obj"
  assert expr.attr.value == "my_attr"

  # Unknown attr
  op_unk: typing.Any = OperationNode(name="sw.getattr", operands=[ValueNode(name="obj")], attributes=[])
  expr_unk: typing.Any = gen._expr_sw_getattr(op_unk)
  assert isinstance(expr_unk, cst.Attribute)
  assert expr_unk.attr.value == "unknown"


def test_expr_sw_call() -> None:
  """Verify expr sw call behavior."""
  gen: typing.Any = DummyGenerator()

  # No operands
  op_err: typing.Any = OperationNode(name="sw.call", operands=[], attributes=[])
  expr_err: typing.Any = gen._expr_sw_call(op_err)
  assert isinstance(expr_err, cst.Call)
  assert getattr(expr_err.func, "value", None) == "unknown"

  # With args and keywords
  op: typing.Any = OperationNode(
    name="sw.call",
    operands=[ValueNode(name="func"), ValueNode(name="arg1"), ValueNode(name="arg2")],
    attributes=[AttributeNode(name="arg_keywords", value='["", "kw2"]')],
  )
  expr: typing.Any = gen._expr_sw_call(op)
  assert isinstance(expr, cst.Call)
  assert isinstance(expr.func, cst.Name)
  assert expr.func.value == "resolved_func"
  assert len(expr.args) == 2
  assert isinstance(expr.args[0].value, cst.Name)
  assert expr.args[0].value.value == "resolved_arg1"
  assert getattr(expr.args[0], "keyword", None) is None
  assert isinstance(expr.args[1].value, cst.Name)
  assert expr.args[1].value.value == "resolved_arg2"
  assert expr.args[1].keyword is not None
  assert expr.args[1].keyword.value == "kw2"


def test_expr_sw_op() -> None:
  """Verify expr sw op behavior."""
  gen: typing.Any = DummyGenerator()

  # Generic call
  op: typing.Any = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="arg1")],
    attributes=[AttributeNode(name="type", value='"np.add"'), AttributeNode(name="arg_keywords", value='[""]')],
  )
  expr: typing.Any = gen._expr_sw_op(op)
  assert isinstance(expr, cst.Call)
  assert isinstance(expr.func, cst.Attribute)  # np.add -> Attribute(Name("np"), Name("add"))
  assert isinstance(expr.func.value, cst.Name)
  assert expr.func.value.value == "np"
  assert expr.func.attr.value == "add"
  assert len(expr.args) == 1
  assert isinstance(expr.args[0].value, cst.Name)
  assert expr.args[0].value.value == "resolved_arg1"

  # binop routing
  op_bin: typing.Any = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="a"), ValueNode(name="b")],
    attributes=[AttributeNode(name="type", value='"binop.add"')],
  )
  expr_bin: typing.Any = gen._expr_sw_op(op_bin)
  assert isinstance(expr_bin, cst.BinaryOperation)
  assert isinstance(expr_bin.operator, cst.Add)


def test_expr_binop() -> None:
  """Verify expr binop behavior."""
  gen: typing.Any = DummyGenerator()

  # Less than 2 operands
  op_err: typing.Any = OperationNode(
    name="sw.op", operands=[ValueNode(name="a")], attributes=[AttributeNode(name="type", value='"binop.add"')]
  )
  expr_err: typing.Any = gen._expr_binop(op_err, '"binop.add"')
  assert isinstance(expr_err, cst.Name)
  assert expr_err.value == "error_binop"

  # All operators
  ops: typing.Any = [
    ("add", cst.Add),
    ("sub", cst.Subtract),
    ("mul", cst.Multiply),
    ("div", cst.Divide),
    ("floordiv", cst.FloorDivide),
    ("mod", cst.Modulo),
    ("pow", cst.Power),
    ("matmul", cst.MatrixMultiply),
    ("lshift", cst.LeftShift),
    ("rshift", cst.RightShift),
    ("and", cst.BitAnd),
    ("or", cst.BitOr),
    ("xor", cst.BitXor),
    ("unknown_op", cst.Add),  # default
  ]
  for op_name, expected_op_cls in ops:
    op: typing.Any = OperationNode(
      name="sw.op",
      operands=[ValueNode(name="a"), ValueNode(name="b")],
      attributes=[AttributeNode(name="type", value=f'"binop.{op_name}"')],
    )
    expr: typing.Any = gen._expr_binop(op, f'"binop.{op_name}"')
    assert isinstance(expr, cst.BinaryOperation)
    assert isinstance(expr.operator, expected_op_cls)
    assert isinstance(expr.left, cst.Name)
    assert expr.left.value == "resolved_a"
    assert isinstance(expr.right, cst.Name)
    assert expr.right.value == "resolved_b"


def test_expr_sw_op_keyword() -> None:
  """Verify expr sw op keyword behavior."""
  gen: typing.Any = DummyGenerator()
  op: typing.Any = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="arg1")],
    attributes=[AttributeNode(name="type", value='"np.add"'), AttributeNode(name="arg_keywords", value='["my_kw"]')],
  )
  expr: typing.Any = gen._expr_sw_op(op)
  assert isinstance(expr, cst.Call)
  assert expr.args[0].keyword is not None
  assert expr.args[0].keyword.value == "my_kw"


def test_parse_keywords_branches_invalid_type() -> None:
  """Verify parse keywords branches invalid type behavior."""
  from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode
  from ml_switcheroo.core.mlir.gen_expressions import ExpressionGeneratorMixin

  class DummyGen(ExpressionGeneratorMixin):
    """Test helper class DummyGen."""

    def _resolve_operand(self, ssa_name: typing.Any) -> typing.Any:
      """Execute resolve operand helper logic.

      Args:
          ssa_name: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return cst.Name("dummy")

  gen: typing.Any = DummyGen()
  # Not a list and not a str
  op: typing.Any = OperationNode(
    name="sw.call", attributes=[AttributeNode(name="arg_keywords", value=123)]
  )  # Intentional wrong type for value
  assert gen._parse_keywords(op) == []


def test_parse_keywords_branches_invalid_list() -> None:
  """Verify parse keywords branches invalid list behavior."""
  gen: typing.Any = DummyGenerator()
  # val is not a list, for example an integer in a string literal
  op: typing.Any = OperationNode(name="sw.call", attributes=[AttributeNode(name="arg_keywords", value='"123"')])
  assert gen._parse_keywords(op) == []
