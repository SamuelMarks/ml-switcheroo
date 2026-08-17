"""Test suite for the Mlir Gen Extra2 module."""

import libcst as cst
from collections import defaultdict
from ml_switcheroo.core.mlir.cst import OperationNode, BlockNode, RegionNode, AttributeNode, ValueNode
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


def test_convert_import():
  """Converts import."""
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
    attributes=[AttributeNode(name="names", value='["sys"]'), AttributeNode(name="aliases", value='["sys"]')],
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
  res_exc = gen._convert_import(op_exc)
  assert isinstance(res_exc.body[0], cst.Pass)


def test_convert_class_def_bases():
  """Converts class def bases."""
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
  """Converts function def annotations."""
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
  """Verifies the behavior of expr sw constant invalid expr."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.constant", operands=[], attributes=[AttributeNode(name="value", value="yield")], results=[], regions=[]
  )
  res = gen._expr_sw_constant(op)
  assert isinstance(res, cst.Name)


def test_convert_class_def_with_body():
  """Converts class def with body."""
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
  """Verifies the behavior of expr sw getattr empty operands."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.getattr", operands=[], attributes=[AttributeNode(name="name", value='"attr"')], results=[], regions=[]
  )
  res = gen._expr_sw_getattr(op)
  assert isinstance(res, cst.Name)
  assert res.value == "error"


def test_stmt_setattr_few_operands():
  """Verifies the behavior of stmt setattr few operands."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.setattr", operands=[], attributes=[], results=[], regions=[])
  res = gen._convert_setattr(op)
  assert isinstance(res.body[0], cst.Pass)


def test_convert_return_no_operands():
  """Converts return no operands."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.return", operands=[], attributes=[], results=[], regions=[])
  res = gen._convert_return(op)
  assert isinstance(res.body[0], cst.Return)
  assert res.body[0].value is None


def test_expr_sw_getattr_happy():
  """Verifies the behavior of expr sw getattr happy."""
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
  """Verifies the behavior of stmt setattr happy."""
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
  """Converts return with operands."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.return", operands=[ValueNode(name="%val")], attributes=[], results=[], regions=[])
  res = gen._convert_return(op)
  assert isinstance(res.body[0], cst.Return)
  assert res.body[0].value is not None
