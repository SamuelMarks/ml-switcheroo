"""Module docstring."""

import pytest
import libcst as cst
from collections import defaultdict

from ml_switcheroo.core.mlir.nodes import OperationNode, BlockNode, RegionNode, AttributeNode, ValueNode, TypeNode
from ml_switcheroo.core.mlir.gen_expressions import ExpressionGeneratorMixin
from ml_switcheroo.core.mlir.gen_statements import StatementGeneratorMixin
from ml_switcheroo.core.mlir.naming import NamingContext


class DummyGenerator(ExpressionGeneratorMixin, StatementGeneratorMixin):
  """Class docstring."""

  def __init__(self):
    """Function docstring."""
    self.ctx = NamingContext()
    self.usage_counts = defaultdict(int)
    self.usage_consumers = {}
    self.resolved_values = {}

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Function docstring."""
    if ssa_name in self.resolved_values:
      return self.resolved_values[ssa_name]
    return cst.Name(f"res_{ssa_name.replace('%', '')}")

  def _convert_block(self, block: BlockNode):
    """Function docstring."""
    if not block.operations:
      return []
    return [cst.SimpleStatementLine(body=[cst.Pass()])]

  def _scan_block_usage(self, block: BlockNode) -> None:
    """Function docstring."""
    pass

  def _create_dotted_name(self, name: str) -> cst.BaseExpression:
    """Function docstring."""
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
    """Function docstring."""
    for a in op.attributes:
      if a.name == attr_name:
        return a.value
    return None


def test_expression_generator_mixin_unimplemented():
  """Function docstring."""

  class IncompleteGen(ExpressionGeneratorMixin):
    """Class docstring."""

    pass

  gen = IncompleteGen()
  with pytest.raises(NotImplementedError):
    gen._resolve_operand("%val")


def test_statement_generator_mixin_unimplemented():
  """Function docstring."""

  class IncompleteGen(StatementGeneratorMixin):
    """Class docstring."""

    pass

  gen = IncompleteGen()
  with pytest.raises(NotImplementedError):
    gen._resolve_operand("%val")
  with pytest.raises(NotImplementedError):
    gen._convert_block(BlockNode(label="^bb0", arguments=[], operations=[]))
  with pytest.raises(NotImplementedError):
    gen._scan_block_usage(BlockNode(label="^bb0", arguments=[], operations=[]))


def test_parse_keywords():
  """Function docstring."""
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
  """Function docstring."""
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
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.getattr", attributes=[], operands=[], results=[], regions=[])
  res = gen._expr_sw_getattr(op)
  assert isinstance(res, cst.Name)
  assert res.value == "error"


def test_expr_sw_call_empty():
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.call", attributes=[], operands=[], results=[], regions=[])
  res = gen._expr_sw_call(op)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Name)
  assert res.func.value == "unknown"


def test_expr_sw_call_with_keywords():
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.call",
    operands=[
      ValueNode(name="%func"),
      ValueNode(name="%arg1"),
      ValueNode(name="%arg2"),
      ValueNode(name="%arg3"),
    ],
    attributes=[AttributeNode(name="arg_keywords", value='["", "", "kw1"]')],
    results=[],
    regions=[],
  )
  res = gen._expr_sw_call(op)
  assert len(res.args) == 3
  assert res.args[0].keyword is None
  assert res.args[2].keyword.value == "kw1"


def test_expr_sw_op():
  """Function docstring."""
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
  """Function docstring."""
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
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.setattr", attributes=[], operands=[], results=[], regions=[])
  res = gen._convert_setattr(op)
  assert isinstance(res.body[0], cst.Pass)


def test_convert_import():
  """Function docstring."""
  gen = DummyGenerator()

  op_star = OperationNode(
    name="sw.import",
    attributes=[
      AttributeNode(name="module", value='"math"'),
      AttributeNode(name="names", value='["*"]'),
      AttributeNode(name="aliases", value='[""]'),
    ],
    operands=[],
    results=[],
    regions=[],
  )
  res_star = gen._convert_import(op_star)
  assert isinstance(res_star.body[0], cst.ImportFrom)
  assert isinstance(res_star.body[0].names, cst.ImportStar)

  op_alias = OperationNode(
    name="sw.import",
    attributes=[
      AttributeNode(name="module", value='"numpy"'),
      AttributeNode(name="names", value='["numpy"]'),
      AttributeNode(name="aliases", value='["np"]'),
    ],
    operands=[],
    results=[],
    regions=[],
  )
  res_alias = gen._convert_import(op_alias)
  assert isinstance(res_alias.body[0], cst.ImportFrom)
  assert res_alias.body[0].names[0].name.value == "numpy"
  assert res_alias.body[0].names[0].asname.name.value == "np"

  op_direct = OperationNode(
    name="sw.import",
    attributes=[
      AttributeNode(name="names", value='["sys"]'),
      AttributeNode(name="aliases", value='["sys"]'),
    ],
    operands=[],
    results=[],
    regions=[],
  )
  res_direct = gen._convert_import(op_direct)
  assert isinstance(res_direct.body[0], cst.Import)
  assert res_direct.body[0].names[0].name.value == "sys"

  op_exc = OperationNode(
    name="sw.import",
    attributes=[
      AttributeNode(name="names", value="invalid syntax"),
      AttributeNode(name="aliases", value="invalid syntax"),
    ],
    operands=[],
    results=[],
    regions=[],
  )
  with pytest.raises(cst.CSTValidationError):
    gen._convert_import(op_exc)


def test_convert_class_def_bases():
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.module",
    attributes=[
      AttributeNode(name="sym_name", value='"MyClass"'),
      AttributeNode(name="bases", value='"nn.Module, object"'),
    ],
    operands=[],
    results=[],
    regions=[],
  )
  res = gen._convert_class_def(op)
  assert res.name.value == "MyClass"
  assert len(res.bases) == 2

  assert isinstance(res.body.body[0].body[0], cst.Pass)


def test_convert_func_def_annotations():
  """Function docstring."""
  gen = DummyGenerator()

  arg1 = ValueNode(name="%arg0")
  arg1_type = type("DummyType", (), {"body": '!sw.type<"torch.Tensor">'})
  arg2 = ValueNode(name="%arg1")
  arg2_type = type("DummyType", (), {"body": '!sw.type<"Any">'})
  arg3 = ValueNode(name="%arg2")
  arg3_type = type("DummyType", (), {"body": ""})

  block = BlockNode(label="^bb0", arguments=[(arg1, arg1_type), (arg2, arg2_type), (arg3, arg3_type)], operations=[])
  region = RegionNode(blocks=[block])

  op = OperationNode(
    name="sw.func",
    attributes=[AttributeNode(name="sym_name", value='"my_func"')],
    operands=[],
    results=[],
    regions=[region],
  )

  res = gen._convert_func_def(op)
  assert res.name.value == "my_func"
  assert len(res.params.params) == 3
  assert res.params.params[0].annotation is not None
  assert res.params.params[1].annotation is None
  assert res.params.params[2].annotation is None

  assert isinstance(res.body.body[0].body[0], cst.Pass)


def test_expr_sw_constant_invalid_expr():
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.constant", operands=[], attributes=[AttributeNode(name="value", value="yield")], results=[], regions=[]
  )
  res = gen._expr_sw_constant(op)
  assert isinstance(res, cst.Name)


def test_convert_class_def_with_body():
  """Function docstring."""
  gen = DummyGenerator()
  block = BlockNode(label="bb0")
  block.operations.append(OperationNode(name="sw.pass", operands=[], attributes=[], results=[], regions=[]))
  op = OperationNode(
    name="sw.class_def",
    operands=[],
    attributes=[AttributeNode(name="sym_name", value='"MyClass"'), AttributeNode(name="bases", value="[]")],
    results=[],
    regions=[RegionNode(blocks=[block])],
  )
  res = gen._convert_class_def(op)
  assert isinstance(res, cst.ClassDef)
  assert len(res.body.body) > 0


def test_expr_sw_getattr_empty_operands():
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.getattr", operands=[], attributes=[AttributeNode(name="name", value='"attr"')], results=[], regions=[]
  )
  res = gen._expr_sw_getattr(op)
  assert isinstance(res, cst.Name)
  assert res.value == "error"


def test_stmt_setattr_few_operands():
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.setattr", operands=[], attributes=[], results=[], regions=[])
  res = gen._convert_setattr(op)
  assert isinstance(res.body[0], cst.Pass)


def test_convert_return_no_operands():
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.return", operands=[], attributes=[], results=[], regions=[])
  res = gen._convert_return(op)
  assert isinstance(res.body[0], cst.Return)
  assert res.body[0].value is None


def test_expr_sw_getattr_happy():
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.getattr",
    operands=[ValueNode(name="%obj")],
    attributes=[AttributeNode(name="name", value='"attr"')],
    results=[],
    regions=[],
  )
  res = gen._expr_sw_getattr(op)
  assert isinstance(res, cst.Attribute)


def test_stmt_setattr_happy():
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.setattr",
    operands=[ValueNode(name="%obj"), ValueNode(name="%val")],
    attributes=[AttributeNode(name="name", value='"attr"')],
    results=[],
    regions=[],
  )
  res = gen._convert_setattr(op)
  assert isinstance(res.body[0], cst.Assign)


def test_convert_return_with_operands():
  """Function docstring."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.return", operands=[ValueNode(name="%val")], attributes=[], results=[], regions=[])
  res = gen._convert_return(op)
  assert isinstance(res.body[0], cst.Return)
  assert res.body[0].value is not None
