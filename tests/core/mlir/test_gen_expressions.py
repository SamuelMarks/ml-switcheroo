"""Test suite for ``MlirExpressionGenerator``."""

from unittest import mock

import libcst as cst

from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode, ValueNode
from ml_switcheroo.core.mlir.gen_expressions import MlirExpressionGenerator


class DummyHost:
  """Typed host implementing ``MlirExprGeneratorHost`` for unit tests."""

  def __init__(self) -> None:
    """Initialize operand resolution table."""
    self._resolved: dict[str, cst.BaseExpression] = {}

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Resolve ``ssa_name`` to a mapped expression or ``resolved_<ssa_name>``.

    Args:
        ssa_name: MLIR SSA identifier string.

    Returns:
        Resolved ``cst.BaseExpression``.
    """
    if ssa_name in self._resolved:
      return self._resolved[ssa_name]
    return cst.Name(f"resolved_{ssa_name}")


def test_parse_keywords() -> None:
  """Verify ``parse_keywords`` handles empty, list, string-encoded list, non-list, and invalid inputs."""
  gen = MlirExpressionGenerator(DummyHost())

  assert gen.parse_keywords(OperationNode(name="op")) == []
  assert gen.parse_keywords(
    OperationNode(name="op", attributes=[AttributeNode(name="arg_keywords", value=['"foo"', "bar"])])
  ) == ["foo", "bar"]
  assert gen.parse_keywords(
    OperationNode(name="op", attributes=[AttributeNode(name="arg_keywords", value='["baz", "qux"]')])
  ) == ["baz", "qux"]
  assert (
    gen.parse_keywords(OperationNode(name="op", attributes=[AttributeNode(name="arg_keywords", value="[invalid")])) == []
  )
  assert (
    gen.parse_keywords(OperationNode(name="sw.call", attributes=[AttributeNode(name="arg_keywords", value=123)])) == []
  )
  assert (
    gen.parse_keywords(OperationNode(name="sw.call", attributes=[AttributeNode(name="arg_keywords", value='"123"')]))
    == []
  )


def test_expr_sw_constant() -> None:
  """Verify ``expr_sw_constant`` parses valid literals, defaults to ``0``, and falls back on syntax errors."""
  gen = MlirExpressionGenerator(DummyHost())

  expr = gen.expr_sw_constant(OperationNode(name="sw.constant", attributes=[AttributeNode(name="value", value="42")]))
  assert isinstance(expr, cst.Integer)
  assert expr.value == "42"

  expr_def = gen.expr_sw_constant(OperationNode(name="sw.constant"))
  assert isinstance(expr_def, cst.Integer)
  assert expr_def.value == "0"

  with mock.patch(
    "libcst.parse_expression", side_effect=cst.ParserSyntaxError("bad", lines=[""], raw_line=1, raw_column=0)
  ):
    expr_inv = gen.expr_sw_constant(
      OperationNode(name="sw.constant", attributes=[AttributeNode(name="value", value="fallback_name")])
    )
    assert isinstance(expr_inv, cst.Name)
    assert expr_inv.value == "fallback_name"


def test_expr_sw_getattr_call_op_and_binop() -> None:
  """Verify ``expr_sw_getattr``, ``expr_sw_call``, ``expr_sw_op``, and ``expr_binop`` outputs."""
  gen = MlirExpressionGenerator(DummyHost())

  expr_err = gen.expr_sw_getattr(OperationNode(name="sw.getattr"))
  assert isinstance(expr_err, cst.Name)
  assert expr_err.value == "error"

  expr_attr = gen.expr_sw_getattr(
    OperationNode(
      name="sw.getattr",
      operands=[ValueNode(name="obj")],
      attributes=[AttributeNode(name="name", value='"my_attr"')],
    )
  )
  assert isinstance(expr_attr, cst.Attribute)
  assert expr_attr.attr.value == "my_attr"

  expr_unk = gen.expr_sw_getattr(OperationNode(name="sw.getattr", operands=[ValueNode(name="obj")]))
  assert isinstance(expr_unk, cst.Attribute)
  assert expr_unk.attr.value == "unknown"

  call_err = gen.expr_sw_call(OperationNode(name="sw.call"))
  assert isinstance(call_err, cst.Call)
  assert isinstance(call_err.func, cst.Name)
  assert call_err.func.value == "unknown"

  call_ok = gen.expr_sw_call(
    OperationNode(
      name="sw.call",
      operands=[ValueNode(name="func"), ValueNode(name="arg1"), ValueNode(name="arg2")],
      attributes=[AttributeNode(name="arg_keywords", value='["", "kw2"]')],
    )
  )
  assert isinstance(call_ok, cst.Call)
  assert len(call_ok.args) == 2
  assert call_ok.args[0].keyword is None
  assert call_ok.args[1].keyword is not None
  assert call_ok.args[1].keyword.value == "kw2"

  op_call = gen.expr_sw_op(
    OperationNode(
      name="sw.op",
      operands=[ValueNode(name="arg1")],
      attributes=[AttributeNode(name="type", value='"np.add"'), AttributeNode(name="arg_keywords", value='["my_kw"]')],
    )
  )
  assert isinstance(op_call, cst.Call)
  assert op_call.args[0].keyword is not None
  assert op_call.args[0].keyword.value == "my_kw"

  op_bin = gen.expr_sw_op(
    OperationNode(
      name="sw.op",
      operands=[ValueNode(name="a"), ValueNode(name="b")],
      attributes=[AttributeNode(name="type", value='"binop.add"')],
    )
  )
  assert isinstance(op_bin, cst.BinaryOperation)
  assert isinstance(op_bin.operator, cst.Add)

  bin_err = gen.expr_binop(OperationNode(name="sw.op", operands=[ValueNode(name="a")]), '"binop.add"')
  assert isinstance(bin_err, cst.Name)
  assert bin_err.value == "error_binop"
