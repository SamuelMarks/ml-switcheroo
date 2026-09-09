"""Test suite for the Mlir Gen Extra module."""

import typing
from collections import defaultdict

import libcst as cst
import pytest

from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, OperationNode, RegionNode, ValueNode
from ml_switcheroo.core.mlir.gen_expressions import ExpressionGeneratorMixin
from ml_switcheroo.core.mlir.gen_statements import StatementGeneratorMixin
from ml_switcheroo.core.mlir.naming import NamingContext


class DummyGenerator(ExpressionGeneratorMixin, StatementGeneratorMixin):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the DummyGenerator instance."""
    self.ctx = NamingContext()
    self.usage_counts: typing.DefaultDict[str, int] = defaultdict(int)
    self.usage_consumers: dict[str, typing.Any] = {}
    self.resolved_values: dict[str, cst.BaseExpression] = {}

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Mock implementation of  resolve operand."""
    if ssa_name in self.resolved_values:
      return self.resolved_values[ssa_name]
    return cst.Name(f"res_{ssa_name.replace('%', '')}")

  def _convert_block(self, block: BlockNode) -> list[cst.BaseStatement]:
    """Mock implementation of  convert block."""
    if not block.operations:
      return []
    return [cst.SimpleStatementLine(body=[cst.Pass()])]

  def _scan_block_usage(self, block: BlockNode) -> None:
    """Mock implementation of  scan block usage."""
    pass

  def _create_dotted_name(self, name: str) -> cst.BaseExpression:
    """Mock implementation of  create dotted name."""
    parts: list[str] = name.split(".")
    if len(parts) == 1:
      return cst.Name(parts[0])
    else:
      attr = cst.Name(parts[-1])
      val: typing.Any = cst.Name(parts[0])
      for p in parts[1:-1]:
        val = cst.Attribute(value=val, attr=cst.Name(p))
      return cst.Attribute(value=val, attr=attr)

  def _get_attr(self, op: OperationNode, attr_name: str) -> typing.Optional[str]:
    """Mock implementation of  get attribute."""
    for a in op.attributes:
      if a.name == attr_name:
        return a.value
    return None


def test_expression_generator_mixin_unimplemented() -> None:
  """Verifies the behavior of expression generator mixin unimplemented."""

  class IncompleteGen(ExpressionGeneratorMixin):
    """Docstring."""

    pass

  gen = IncompleteGen()
  with pytest.raises(NotImplementedError):
    gen._resolve_operand("%val")


def test_statement_generator_mixin_unimplemented() -> None:
  """Verifies the behavior of statement generator mixin unimplemented."""

  class IncompleteGen(StatementGeneratorMixin):
    """Docstring."""

    pass

  gen = IncompleteGen()
  with pytest.raises(NotImplementedError):
    gen._resolve_operand("%val")
  with pytest.raises(NotImplementedError):
    gen._convert_block(BlockNode(label="^bb0", arguments=[], operations=[]))
  with pytest.raises(NotImplementedError):
    gen._scan_block_usage(BlockNode(label="^bb0", arguments=[], operations=[]))


def test_parse_keywords() -> None:
  """Parses keywords."""
  gen = DummyGenerator()
  op1 = OperationNode(
    name="sw.call",
    attributes=[AttributeNode(name="arg_keywords", value='["arg1", "arg2"]')],
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


def test_expr_sw_constant_exception() -> None:
  """Verifies the behavior of expr sw constant correctly handling an exception."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.constant",
    attributes=[AttributeNode(name="value", value="invalid_syntax")],
    operands=[],
    results=[],
    regions=[],
  )
  res: typing.Any = gen._expr_sw_constant(op)
  assert isinstance(res, cst.Name)
  assert res.value == "invalid_syntax"


def test_expr_sw_getattr_empty() -> None:
  """Verifies the behavior of expr sw getattr empty."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.getattr", attributes=[], operands=[], results=[], regions=[])
  res: typing.Any = gen._expr_sw_getattr(op)
  assert isinstance(res, cst.Name)
  assert res.value == "error"


def test_expr_sw_call_empty() -> None:
  """Verifies the behavior of expr sw call empty."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.call", attributes=[], operands=[], results=[], regions=[])
  res: typing.Any = gen._expr_sw_call(op)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Name)
  assert res.func.value == "unknown"


def test_expr_sw_call_with_keywords() -> None:
  """Verifies the behavior of expr sw call with keywords."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.call",
    operands=[ValueNode(name="%func"), ValueNode(name="%arg1"), ValueNode(name="%arg2"), ValueNode(name="%arg3")],
    attributes=[AttributeNode(name="arg_keywords", value='["", "", "kw1"]')],
    results=[],
    regions=[],
  )
  res: typing.Any = gen._expr_sw_call(op)
  assert len(res.args) == 3
  assert res.args[0].keyword is None
  assert res.args[2].keyword.value == "kw1"


def test_expr_sw_op() -> None:
  """Verifies the behavior of expr sw op."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="%arg1")],
    attributes=[AttributeNode(name="type", value='"torch.add"'), AttributeNode(name="arg_keywords", value='["kw1"]')],
    results=[],
    regions=[],
  )
  res: typing.Any = gen._expr_sw_op(op)
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
  res2: typing.Any = gen._expr_sw_op(op2)
  assert isinstance(res2, cst.BinaryOperation)
  assert isinstance(res2.operator, cst.Add)


def test_expr_binop_all_ops() -> None:
  """Verifies the behavior of expr binop all ops."""
  gen = DummyGenerator()
  op_err = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="%1")],
    attributes=[AttributeNode(name="type", value='"binop.add"')],
    results=[],
    regions=[],
  )
  res_err: typing.Any = gen._expr_binop(op_err, "binop.add")
  assert isinstance(res_err, cst.Name)
  assert res_err.value == "error_binop"
  ops: dict[str, typing.Any] = {
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
    res: typing.Any = gen._expr_binop(op, f"binop.{op_name}")
    assert isinstance(res.operator, expected_cst_op)


def test_convert_setattr_empty() -> None:
  """Converts setattr empty."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.setattr", attributes=[], operands=[], results=[], regions=[])
  res: typing.Any = gen._convert_setattr(op)
  assert isinstance(res.body[0], cst.Pass)


# --- Merged from test_mlir_gen_extra2.py ---


class DummyGeneratorExtra(ExpressionGeneratorMixin, StatementGeneratorMixin):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the DummyGenerator instance."""
    self.ctx = NamingContext()
    self.usage_counts: typing.DefaultDict[str, int] = defaultdict(int)
    self.usage_consumers: dict[str, typing.Any] = {}
    self.resolved_values: dict[str, cst.BaseExpression] = {}

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Mock implementation of  resolve operand."""
    if ssa_name in self.resolved_values:
      return self.resolved_values[ssa_name]
    return cst.Name(f"res_{ssa_name.replace('%', '')}")

  def _convert_block(self, block: BlockNode) -> list[cst.BaseStatement]:
    """Mock implementation of  convert block."""
    if not block.operations:
      return []
    return [cst.SimpleStatementLine(body=[cst.Pass()])]

  def _scan_block_usage(self, block: BlockNode) -> None:
    """Mock implementation of  scan block usage."""
    pass

  def _create_dotted_name(self, name: str) -> cst.BaseExpression:
    """Mock implementation of  create dotted name."""
    parts: list[str] = name.split(".")
    if len(parts) == 1:
      return cst.Name(parts[0])
    else:
      attr = cst.Name(parts[-1])
      val: typing.Any = cst.Name(parts[0])
      for p in parts[1:-1]:
        val = cst.Attribute(value=val, attr=cst.Name(p))
      return cst.Attribute(value=val, attr=attr)

  def _get_attr(self, op: OperationNode, attr_name: str) -> typing.Optional[str]:
    """Mock implementation of  get attribute."""
    for a in op.attributes:
      if a.name == attr_name:
        return a.value
    return None


def test_convert_import() -> None:
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
  res_star: typing.Any = gen._convert_import(op_star)
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
  res_alias: typing.Any = gen._convert_import(op_alias)
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
  res_direct: typing.Any = gen._convert_import(op_direct)
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
  res_exc: typing.Any = gen._convert_import(op_exc)
  assert isinstance(res_exc.body[0], cst.Pass)


def test_convert_class_def_bases() -> None:
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
  res: typing.Any = gen._convert_class_def(op)
  assert res.name.value == "MyClass"
  assert len(res.bases) == 2
  assert isinstance(res.body.body[0].body[0], cst.Pass)


def test_convert_func_def_annotations() -> None:
  """Converts function def annotations."""
  gen = DummyGenerator()
  arg1 = ValueNode(name="%arg0")
  arg1_type: typing.Any = type("DummyType", (), {"body": '!sw.type<"torch.Tensor">'})
  arg2 = ValueNode(name="%arg1")
  arg2_type: typing.Any = type("DummyType", (), {"body": '!sw.type<"Any">'})
  arg3 = ValueNode(name="%arg2")
  arg3_type: typing.Any = type("DummyType", (), {"body": ""})
  block = BlockNode(label="^bb0", arguments=[(arg1, arg1_type), (arg2, arg2_type), (arg3, arg3_type)], operations=[])  # type: ignore
  region = RegionNode(blocks=[block])
  op = OperationNode(
    name="sw.func",
    attributes=[AttributeNode(name="sym_name", value='"my_func"')],
    operands=[],
    results=[],
    regions=[region],
  )
  res: typing.Any = gen._convert_func_def(op)
  assert res.name.value == "my_func"
  assert len(res.params.params) == 3
  assert res.params.params[0].annotation is not None
  assert res.params.params[1].annotation is None
  assert res.params.params[2].annotation is None
  assert isinstance(res.body.body[0].body[0], cst.Pass)


def test_expr_sw_constant_invalid_expr() -> None:
  """Verifies the behavior of expr sw constant invalid expr."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.constant", operands=[], attributes=[AttributeNode(name="value", value="yield")], results=[], regions=[]
  )
  res: typing.Any = gen._expr_sw_constant(op)
  assert isinstance(res, cst.Name)


def test_convert_class_def_with_body() -> None:
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
  res: typing.Any = gen._convert_class_def(op)
  assert isinstance(res, cst.ClassDef)
  assert len(res.body.body) > 0


def test_expr_sw_getattr_empty_operands() -> None:
  """Verifies the behavior of expr sw getattr empty operands."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.getattr", operands=[], attributes=[AttributeNode(name="name", value='"attr"')], results=[], regions=[]
  )
  res: typing.Any = gen._expr_sw_getattr(op)
  assert isinstance(res, cst.Name)
  assert res.value == "error"


def test_stmt_setattr_few_operands() -> None:
  """Verifies the behavior of stmt setattr few operands."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.setattr", operands=[], attributes=[], results=[], regions=[])
  res: typing.Any = gen._convert_setattr(op)
  assert isinstance(res.body[0], cst.Pass)


def test_convert_return_no_operands() -> None:
  """Converts return no operands."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.return", operands=[], attributes=[], results=[], regions=[])
  res: typing.Any = gen._convert_return(op)
  assert isinstance(res.body[0], cst.Return)
  assert res.body[0].value is None


def test_expr_sw_getattr_happy() -> None:
  """Verifies the behavior of expr sw getattr happy."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.getattr",
    operands=[ValueNode(name="%obj")],
    attributes=[AttributeNode(name="name", value='"attr"')],
    results=[],
    regions=[],
  )
  res: typing.Any = gen._expr_sw_getattr(op)
  assert isinstance(res, cst.Attribute)


def test_stmt_setattr_happy() -> None:
  """Verifies the behavior of stmt setattr happy."""
  gen = DummyGenerator()
  op = OperationNode(
    name="sw.setattr",
    operands=[ValueNode(name="%obj"), ValueNode(name="%val")],
    attributes=[AttributeNode(name="name", value='"attr"')],
    results=[],
    regions=[],
  )
  res: typing.Any = gen._convert_setattr(op)
  assert isinstance(res.body[0], cst.Assign)


def test_convert_return_with_operands() -> None:
  """Converts return with operands."""
  gen = DummyGenerator()
  op = OperationNode(name="sw.return", operands=[ValueNode(name="%val")], attributes=[], results=[], regions=[])
  res: typing.Any = gen._convert_return(op)
  assert isinstance(res.body[0], cst.Return)
  assert res.body[0].value is not None


def test_gen_expressions_and_statements_missing_branches() -> None:
  """Test edge cases and branches in expression and statement generation."""
  from ml_switcheroo.core.mlir.cst import TypeNode

  gen = DummyGenerator()

  # 1. _parse_keywords with list value (line 50)
  op_list = OperationNode(
    name="sw.op",
    operands=[],
    attributes=[AttributeNode(name="arg_keywords", value=['"k1"', '"k2"'])],
    results=[],
    regions=[],
  )
  assert gen._parse_keywords(op_list) == ["k1", "k2"]

  # 2. _parse_keywords with non-string/non-list value and subsequent attribute (54->46)
  op_non_list = OperationNode(
    name="sw.op",
    operands=[],
    attributes=[
      AttributeNode(name="arg_keywords", value=typing.cast(typing.Any, 123)),
      AttributeNode(name="other", value="ignored"),
    ],
    results=[],
    regions=[],
  )
  assert gen._parse_keywords(op_non_list) == []

  # 3. _expr_sw_op with empty keyword string (166->169)
  op_call = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="%x"), ValueNode(name="%y")],
    attributes=[
      AttributeNode(name="type", value='"torch.add"'),
      AttributeNode(name="arg_keywords", value='["", "other"]'),
    ],
    results=[],
    regions=[],
  )
  call_node = gen._expr_sw_op(op_call)
  assert isinstance(call_node, cst.Call)
  assert call_node.args[0].keyword is None
  assert call_node.args[1].keyword is not None

  # 4. _convert_import with missing names/aliases and empty fallback (103->105, 105->110)
  op_empty_import = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[],
    results=[],
    regions=[],
  )
  pass_stmt = gen._convert_import(op_empty_import)
  assert isinstance(pass_stmt.body[0], cst.Pass)

  # 4b. _convert_import with module_val set but empty import_aliases (136)
  op_mod_import = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[AttributeNode(name="module", value='"os"')],
    results=[],
    regions=[],
  )
  import_stmt = gen._convert_import(op_mod_import)
  assert isinstance(import_stmt.body[0], cst.Import)

  # 5. _convert_class_def with empty elements in bases_attr (179->177)
  op_class = OperationNode(
    name="sw.class",
    operands=[],
    attributes=[
      AttributeNode(name="sym_name", value='"MyClass"'),
      AttributeNode(name="bases", value="[, ,]"),
    ],
    results=[],
    regions=[],
  )
  class_node = gen._convert_class_def(op_class)
  assert isinstance(class_node, cst.ClassDef)
  assert len(class_node.bases) == 0

  # 6. _convert_func_def with argument type not starting with !sw.type<
  arg_block = BlockNode(
    label="^bb0",
    arguments=[(ValueNode(name="%x"), TypeNode(body="tensor<f32>"))],
    operations=[],
  )
  op_func = OperationNode(
    name="sw.func",
    operands=[],
    attributes=[AttributeNode(name="sym_name", value='"my_func"')],
    results=[],
    regions=[RegionNode(blocks=[arg_block])],
  )
  func_node = gen._convert_func_def(op_func)
  assert isinstance(func_node, cst.FunctionDef)
  assert func_node.params.params[0].annotation is None

  # 6b. _convert_func_def with empty regions (219->237)
  op_empty_func = OperationNode(
    name="sw.func",
    operands=[],
    attributes=[AttributeNode(name="sym_name", value='"empty_func"')],
    results=[],
    regions=[],
  )
  func_empty_node = gen._convert_func_def(op_empty_func)
  assert isinstance(func_empty_node, cst.FunctionDef)
