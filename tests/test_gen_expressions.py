"""Module docstring."""

import pytest
import libcst as cst
from unittest import mock
from typing import Dict, List

from ml_switcheroo.core.mlir.gen_expressions import ExpressionGeneratorMixin
from ml_switcheroo.core.mlir.cst import OperationNode, AttributeNode, ValueNode


class DummyGenerator(ExpressionGeneratorMixin):
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self._resolved: Dict[str, cst.BaseExpression] = {}

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Function doc."""
    if ssa_name in self._resolved:
      return self._resolved[ssa_name]
    return cst.Name(f"resolved_{ssa_name}")


def test_resolve_operand_not_implemented() -> None:
  """Docstring."""

  class IncompleteGenerator(ExpressionGeneratorMixin):
    """Class doc."""

    pass

  gen: IncompleteGenerator = IncompleteGenerator()
  with pytest.raises(NotImplementedError):
    gen._resolve_operand("foo")


def test_parse_keywords() -> None:
  """Docstring."""
  gen: DummyGenerator = DummyGenerator()

  # Empty
  op: OperationNode = OperationNode(name="op", operands=[], attributes=[])
  assert gen._parse_keywords(op) == []

  # List case
  op_list: OperationNode = OperationNode(
    name="op", operands=[], attributes=[AttributeNode(name="arg_keywords", value=['"foo"', "bar"])]
  )
  assert gen._parse_keywords(op_list) == ["foo", "bar"]

  # String case
  op_str: OperationNode = OperationNode(
    name="op", operands=[], attributes=[AttributeNode(name="arg_keywords", value='["baz", "qux"]')]
  )
  assert gen._parse_keywords(op_str) == ["baz", "qux"]

  # Invalid string case
  op_inv: OperationNode = OperationNode(
    name="op", operands=[], attributes=[AttributeNode(name="arg_keywords", value="[invalid")]
  )
  assert gen._parse_keywords(op_inv) == []


def test_expr_sw_constant() -> None:
  """Docstring."""
  gen: DummyGenerator = DummyGenerator()

  # Valid parseable expression
  op: OperationNode = OperationNode(name="sw.constant", operands=[], attributes=[AttributeNode(name="value", value="42")])
  expr: cst.BaseExpression = gen._expr_sw_constant(op)
  assert isinstance(expr, cst.Integer)
  assert expr.value == "42"

  # Default value (0)
  op_def: OperationNode = OperationNode(name="sw.constant", operands=[], attributes=[])
  expr_def: cst.BaseExpression = gen._expr_sw_constant(op_def)
  assert isinstance(expr_def, cst.Integer)
  assert expr_def.value == "0"

  # Unparseable expression fallback to Name
  with mock.patch("libcst.parse_expression", side_effect=Exception("Failed")):
    op_inv: OperationNode = OperationNode(
      name="sw.constant", operands=[], attributes=[AttributeNode(name="value", value="fallback_name")]
    )
    expr_inv: cst.BaseExpression = gen._expr_sw_constant(op_inv)
    assert isinstance(expr_inv, cst.Name)
    assert expr_inv.value == "fallback_name"


def test_expr_sw_getattr() -> None:
  """Docstring."""
  gen: DummyGenerator = DummyGenerator()

  # No operands error
  op_err: OperationNode = OperationNode(name="sw.getattr", operands=[], attributes=[])
  expr_err: cst.BaseExpression = gen._expr_sw_getattr(op_err)
  assert isinstance(expr_err, cst.Name)
  assert expr_err.value == "error"

  # Success
  op: OperationNode = OperationNode(
    name="sw.getattr", operands=[ValueNode(name="obj")], attributes=[AttributeNode(name="name", value='"my_attr"')]
  )
  expr: cst.BaseExpression = gen._expr_sw_getattr(op)
  assert isinstance(expr, cst.Attribute)
  assert isinstance(expr.value, cst.Name)
  assert expr.value.value == "resolved_obj"
  assert expr.attr.value == "my_attr"

  # Unknown attr
  op_unk: OperationNode = OperationNode(name="sw.getattr", operands=[ValueNode(name="obj")], attributes=[])
  expr_unk: cst.BaseExpression = gen._expr_sw_getattr(op_unk)
  assert isinstance(expr_unk, cst.Attribute)
  assert expr_unk.attr.value == "unknown"


def test_expr_sw_call() -> None:
  """Docstring."""
  gen: DummyGenerator = DummyGenerator()

  # No operands
  op_err: OperationNode = OperationNode(name="sw.call", operands=[], attributes=[])
  expr_err: cst.BaseExpression = gen._expr_sw_call(op_err)
  assert isinstance(expr_err, cst.Call)
  assert getattr(expr_err.func, "value", None) == "unknown"

  # With args and keywords
  op: OperationNode = OperationNode(
    name="sw.call",
    operands=[ValueNode(name="func"), ValueNode(name="arg1"), ValueNode(name="arg2")],
    attributes=[AttributeNode(name="arg_keywords", value='["", "kw2"]')],
  )
  expr: cst.BaseExpression = gen._expr_sw_call(op)
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
  """Docstring."""
  gen: DummyGenerator = DummyGenerator()

  # Generic call
  op: OperationNode = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="arg1")],
    attributes=[AttributeNode(name="type", value='"np.add"'), AttributeNode(name="arg_keywords", value='[""]')],
  )
  expr: cst.BaseExpression = gen._expr_sw_op(op)
  assert isinstance(expr, cst.Call)
  assert isinstance(expr.func, cst.Attribute)  # np.add -> Attribute(Name("np"), Name("add"))
  assert isinstance(expr.func.value, cst.Name)
  assert expr.func.value.value == "np"
  assert expr.func.attr.value == "add"
  assert len(expr.args) == 1
  assert isinstance(expr.args[0].value, cst.Name)
  assert expr.args[0].value.value == "resolved_arg1"

  # binop routing
  op_bin: OperationNode = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="a"), ValueNode(name="b")],
    attributes=[AttributeNode(name="type", value='"binop.add"')],
  )
  expr_bin: cst.BaseExpression = gen._expr_sw_op(op_bin)
  assert isinstance(expr_bin, cst.BinaryOperation)
  assert isinstance(expr_bin.operator, cst.Add)


def test_expr_binop() -> None:
  """Docstring."""
  gen: DummyGenerator = DummyGenerator()

  # Less than 2 operands
  op_err: OperationNode = OperationNode(
    name="sw.op", operands=[ValueNode(name="a")], attributes=[AttributeNode(name="type", value='"binop.add"')]
  )
  expr_err: cst.BaseExpression = gen._expr_binop(op_err, '"binop.add"')
  assert isinstance(expr_err, cst.Name)
  assert expr_err.value == "error_binop"

  # All operators
  ops: List[tuple[str, type]] = [
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
    op: OperationNode = OperationNode(
      name="sw.op",
      operands=[ValueNode(name="a"), ValueNode(name="b")],
      attributes=[AttributeNode(name="type", value=f'"binop.{op_name}"')],
    )
    expr: cst.BaseExpression = gen._expr_binop(op, f'"binop.{op_name}"')
    assert isinstance(expr, cst.BinaryOperation)
    assert isinstance(expr.operator, expected_op_cls)
    assert isinstance(expr.left, cst.Name)
    assert expr.left.value == "resolved_a"
    assert isinstance(expr.right, cst.Name)
    assert expr.right.value == "resolved_b"


def test_expr_sw_op_keyword() -> None:
  """Function doc."""
  gen: DummyGenerator = DummyGenerator()
  op: OperationNode = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="arg1")],
    attributes=[AttributeNode(name="type", value='"np.add"'), AttributeNode(name="arg_keywords", value='["my_kw"]')],
  )
  expr: cst.BaseExpression = gen._expr_sw_op(op)
  assert isinstance(expr, cst.Call)
  assert expr.args[0].keyword is not None
  assert expr.args[0].keyword.value == "my_kw"


def test_parse_keywords_branches_invalid_type() -> None:
  """Function doc."""
  from ml_switcheroo.core.mlir.gen_expressions import ExpressionGeneratorMixin
  from ml_switcheroo.core.mlir.cst import OperationNode, AttributeNode

  class DummyGen(ExpressionGeneratorMixin):
    """Class doc."""

    def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
      """Function doc."""
      return cst.Name("dummy")

  gen: DummyGen = DummyGen()
  # Not a list and not a str
  op: OperationNode = OperationNode(
    name="sw.call", attributes=[AttributeNode(name="arg_keywords", value=123)]
  )  # Intentional wrong type for value
  assert gen._parse_keywords(op) == []


def test_parse_keywords_branches_invalid_list() -> None:
  """Function doc."""
  gen: DummyGenerator = DummyGenerator()
  # val is not a list, for example an integer in a string literal
  op: OperationNode = OperationNode(name="sw.call", attributes=[AttributeNode(name="arg_keywords", value='"123"')])
  assert gen._parse_keywords(op) == []
