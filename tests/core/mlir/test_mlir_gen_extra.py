"""Test suite for the Mlir Gen Extra module."""

import pytest
import libcst as cst
from collections import defaultdict
from ml_switcheroo.core.mlir.nodes import OperationNode, BlockNode, AttributeNode, ValueNode
from ml_switcheroo.core.mlir.gen_expressions import ExpressionGeneratorMixin
from ml_switcheroo.core.mlir.gen_statements import StatementGeneratorMixin
from ml_switcheroo.core.mlir.naming import NamingContext


class DummyGenerator(ExpressionGeneratorMixin, StatementGeneratorMixin):
  """Dummy Generator class for testing purposes."""

  def __init__(self):
    """Initializes the DummyGenerator instance."""
    self.ctx = NamingContext()
    self.usage_counts = defaultdict(int)
    self.usage_consumers = {}
    self.resolved_values = {}

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Mock implementation of  resolve operand."""
    if ssa_name in self.resolved_values:
      return self.resolved_values[ssa_name]
    return cst.Name(f"res_{ssa_name.replace('%', '')}")

  def _convert_block(self, block: BlockNode):
    """Mock implementation of  convert block."""
    if not block.operations:
      return []
    return [cst.SimpleStatementLine(body=[cst.Pass()])]

  def _scan_block_usage(self, block: BlockNode) -> None:
    """Mock implementation of  scan block usage."""
    pass

  def _create_dotted_name(self, name: str) -> cst.BaseExpression:
    """Mock implementation of  create dotted name."""
    parts = name.split(".")
    if len(parts) == 1:
      return cst.Name(parts[0])
    else:
      attr = cst.Name(parts[-1])
      val = cst.Name(parts[0])
      for p in parts[1:-1]:
        val = cst.Attribute(value=val, attr=cst.Name(p))
      return cst.Attribute(value=val, attr=attr)

  def _get_attr(self, op: OperationNode, attr_name: str) -> str:
    """Mock implementation of  get attribute."""
    for a in op.attributes:
      if a.name == attr_name:
        return a.value
    return None


def test_expression_generator_mixin_unimplemented():
  """Verifies the behavior of expression generator mixin unimplemented."""

  class IncompleteGen(ExpressionGeneratorMixin):
    """Test suite for the Incomplete Gen component."""

    pass

  gen = IncompleteGen()
  with pytest.raises(NotImplementedError):
    gen._resolve_operand("%val")


def test_statement_generator_mixin_unimplemented():
  """Verifies the behavior of statement generator mixin unimplemented."""

  class IncompleteGen(StatementGeneratorMixin):
    """Test suite for the Incomplete Gen component."""

    pass

  gen = IncompleteGen()
  with pytest.raises(NotImplementedError):
    gen._resolve_operand("%val")
  with pytest.raises(NotImplementedError):
    gen._convert_block(BlockNode(label="^bb0", arguments=[], operations=[]))
  with pytest.raises(NotImplementedError):
    gen._scan_block_usage(BlockNode(label="^bb0", arguments=[], operations=[]))


def test_parse_keywords():
  """Parses keywords."""
  gen = DummyGenerator()
  op1 = OperationNode(
    name="sw.call",
    attributes=[AttributeNode(name="arg_keywords", value=['"arg1"', '"arg2"'])],
    operands=[],
    results=[],
    regions=[],
  )
  assert gen._parse_keywords(op1) == ["arg1", "arg2"]
  op2 = OperationNode(
    name="sw.call",
    attributes=[AttributeNode(name="arg_keywords", value='["arg1", "arg2"]')],
    operands=[],
    results=[],
    regions=[],
  )
  assert gen._parse_keywords(op2) == ["arg1", "arg2"]
  op3 = OperationNode(
    name="sw.call", attributes=[AttributeNode(name="arg_keywords", value="[invalid")], operands=[], results=[], regions=[]
  )
  assert gen._parse_keywords(op3) == []
  op4 = OperationNode(
    name="sw.call",
    attributes=[AttributeNode(name="arg_keywords", value='"not a list"')],
    operands=[],
    results=[],
    regions=[],
  )
  assert gen._parse_keywords(op4) == []


def test_expr_sw_constant_exception():
  """Verifies the behavior of expr sw constant correctly handling an exception."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.constant",
    attributes=[AttributeNode(name="value", value="invalid_syntax")],
    operands=[],
    results=[],
    regions=[],
  )
  res = gen._expr_sw_constant(op)
  assert isinstance(res, cst.Name)
  assert res.value == "invalid_syntax"


def test_expr_sw_getattr_empty():
  """Verifies the behavior of expr sw getattr empty."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.getattr", attributes=[], operands=[], results=[], regions=[])
  res = gen._expr_sw_getattr(op)
  assert isinstance(res, cst.Name)
  assert res.value == "error"


def test_expr_sw_call_empty():
  """Verifies the behavior of expr sw call empty."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.call", attributes=[], operands=[], results=[], regions=[])
  res = gen._expr_sw_call(op)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Name)
  assert res.func.value == "unknown"


def test_expr_sw_call_with_keywords():
  """Verifies the behavior of expr sw call with keywords."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.call",
    operands=[ValueNode(name="%func"), ValueNode(name="%arg1"), ValueNode(name="%arg2"), ValueNode(name="%arg3")],
    attributes=[AttributeNode(name="arg_keywords", value='["", "", "kw1"]')],
    results=[],
    regions=[],
  )
  res = gen._expr_sw_call(op)
  assert len(res.args) == 3
  assert res.args[0].keyword is None
  assert res.args[2].keyword.value == "kw1"


def test_expr_sw_op():
  """Verifies the behavior of expr sw op."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="%arg1")],
    attributes=[AttributeNode(name="type", value='"torch.add"'), AttributeNode(name="arg_keywords", value='["kw1"]')],
    results=[],
    regions=[],
  )
  res = gen._expr_sw_op(op)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 1
  assert res.args[0].keyword.value == "kw1"
  op2 = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="%1"), ValueNode(name="%2")],
    attributes=[AttributeNode(name="type", value='"binop.add"')],
    results=[],
    regions=[],
  )
  res2 = gen._expr_sw_op(op2)
  assert isinstance(res2, cst.BinaryOperation)
  assert isinstance(res2.operator, cst.Add)


def test_expr_binop_all_ops():
  """Verifies the behavior of expr binop all ops."""
  gen = DummyGenerator()
  op_err = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="%1")],
    attributes=[AttributeNode(name="type", value='"binop.add"')],
    results=[],
    regions=[],
  )
  res_err = gen._expr_binop(op_err, "binop.add")
  assert isinstance(res_err, cst.Name)
  assert res_err.value == "error_binop"
  ops = {
    "add": cst.Add,
    "sub": cst.Subtract,
    "mul": cst.Multiply,
    "div": cst.Divide,
    "floordiv": cst.FloorDivide,
    "mod": cst.Modulo,
    "pow": cst.Power,
    "matmul": cst.MatrixMultiply,
    "lshift": cst.LeftShift,
    "rshift": cst.RightShift,
    "and": cst.BitAnd,
    "or": cst.BitOr,
    "xor": cst.BitXor,
    "unknown_op": cst.Add,
  }
  for op_name, expected_cst_op in ops.items():
    op = OperationNode(
      name="sw.op",
      operands=[ValueNode(name="%1"), ValueNode(name="%2")],
      attributes=[AttributeNode(name="type", value=f'"binop.{op_name}"')],
      results=[],
      regions=[],
    )
    res = gen._expr_binop(op, f"binop.{op_name}")
    assert isinstance(res.operator, expected_cst_op)


def test_convert_setattr_empty():
  """Converts setattr empty."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.setattr", attributes=[], operands=[], results=[], regions=[])
  res = gen._convert_setattr(op)
  assert isinstance(res.body[0], cst.Pass)
