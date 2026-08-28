"""Module docstring."""

import pytest
import libcst as cst
from typing import List, Dict, Any
from collections import defaultdict

from ml_switcheroo.core.mlir.gen_statements import StatementGeneratorMixin
from ml_switcheroo.core.mlir.cst import OperationNode, AttributeNode, ValueNode, BlockNode, RegionNode, TypeNode
from ml_switcheroo.core.mlir.naming import NamingContext


class DummyGenerator(StatementGeneratorMixin):
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.ctx: NamingContext = NamingContext()
    self.usage_counts: Dict[str, int] = defaultdict(int)
    self.usage_consumers: Dict[str, Any] = {}
    self._resolved: Dict[str, cst.BaseExpression] = {}

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Function doc."""
    if ssa_name in self._resolved:
      return self._resolved[ssa_name]
    return cst.Name(f"resolved_{ssa_name}")

  def _convert_block(self, block: BlockNode) -> List[cst.BaseStatement]:
    """Function doc."""
    return [cst.SimpleStatementLine(body=[cst.Pass()])]

  def _scan_block_usage(self, block: BlockNode) -> None:
    """Function doc."""
    pass


def test_resolve_operand_not_implemented() -> None:
  """Docstring."""

  class IncompleteGenerator(StatementGeneratorMixin):
    """Class doc."""

    def _convert_block(self, block: BlockNode) -> List[cst.BaseStatement]:
      """Function doc."""
      return []

    def _scan_block_usage(self, block: BlockNode) -> None:
      """Function doc."""
      pass

  gen: IncompleteGenerator = IncompleteGenerator()
  with pytest.raises(NotImplementedError):
    gen._resolve_operand("foo")


def test_convert_block_not_implemented() -> None:
  """Docstring."""

  class IncompleteGenerator(StatementGeneratorMixin):
    """Class doc."""

    def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
      """Function doc."""
      return cst.Name("dummy")

    def _scan_block_usage(self, block: BlockNode) -> None:
      """Function doc."""
      pass

  gen: IncompleteGenerator = IncompleteGenerator()
  with pytest.raises(NotImplementedError):
    gen._convert_block(BlockNode(label="foo"))


def test_scan_block_usage_not_implemented() -> None:
  """Docstring."""

  class IncompleteGenerator(StatementGeneratorMixin):
    """Class doc."""

    def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
      """Function doc."""
      return cst.Name("dummy")

    def _convert_block(self, block: BlockNode) -> List[cst.BaseStatement]:
      """Function doc."""
      return []

  gen: IncompleteGenerator = IncompleteGenerator()
  with pytest.raises(NotImplementedError):
    gen._scan_block_usage(BlockNode(label="foo"))


def test_convert_setattr() -> None:
  """Docstring."""
  gen: DummyGenerator = DummyGenerator()

  # Missing operands
  op_err: OperationNode = OperationNode(name="sw.setattr", operands=[], attributes=[])
  stmt_err: cst.BaseStatement = gen._convert_setattr(op_err)
  assert isinstance(stmt_err, cst.SimpleStatementLine)
  assert isinstance(stmt_err.body[0], cst.Pass)

  # Valid
  op: OperationNode = OperationNode(
    name="sw.setattr",
    operands=[ValueNode(name="obj"), ValueNode(name="val")],
    attributes=[AttributeNode(name="name", value='"my_attr"')],
  )
  stmt: cst.BaseStatement = gen._convert_setattr(op)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assign = stmt.body[0]
  assert isinstance(assign, cst.Assign)
  assert isinstance(assign.targets[0].target, cst.Attribute)
  assert isinstance(assign.targets[0].target.value, cst.Name)
  assert assign.targets[0].target.value.value == "resolved_obj"
  assert assign.targets[0].target.attr.value == "my_attr"
  assert isinstance(assign.value, cst.Name)
  assert assign.value.value == "resolved_val"


def test_convert_import() -> None:
  """Docstring."""
  gen: DummyGenerator = DummyGenerator()

  # From import multiple names and aliases
  op_from: OperationNode = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[
      AttributeNode(name="module", value='"os.path"'),
      AttributeNode(name="names", value="['join', 'exists']"),
      AttributeNode(name="aliases", value="['join', 'path_exists']"),
    ],
  )
  stmt: cst.BaseStatement = gen._convert_import(op_from)
  assert isinstance(stmt, cst.SimpleStatementLine)
  import_node = stmt.body[0]
  assert isinstance(import_node, cst.ImportFrom)
  assert isinstance(import_node.module, cst.Attribute)
  assert isinstance(import_node.module.value, cst.Name)
  assert import_node.module.value.value == "os"
  assert import_node.module.attr.value == "path"
  assert isinstance(import_node.names, tuple) or isinstance(import_node.names, list)
  assert len(import_node.names) == 2
  assert import_node.names[0].name.value == "join"
  assert import_node.names[0].asname is None
  assert import_node.names[1].name.value == "exists"
  assert import_node.names[1].asname is not None
  assert import_node.names[1].asname.name.value == "path_exists"

  # Import star
  op_star: OperationNode = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[
      AttributeNode(name="module", value='"os"'),
      AttributeNode(name="names", value="['*']"),
      AttributeNode(name="aliases", value="['*']"),
    ],
  )
  stmt_star: cst.BaseStatement = gen._convert_import(op_star)
  assert isinstance(stmt_star, cst.SimpleStatementLine)
  assert isinstance(stmt_star.body[0], cst.ImportFrom)
  assert isinstance(stmt_star.body[0].names, cst.ImportStar)

  # Import normal
  op_norm: OperationNode = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[AttributeNode(name="names", value="['sys']"), AttributeNode(name="aliases", value="['sys']")],
  )
  stmt_norm: cst.BaseStatement = gen._convert_import(op_norm)
  assert isinstance(stmt_norm, cst.SimpleStatementLine)
  assert isinstance(stmt_norm.body[0], cst.Import)
  assert isinstance(stmt_norm.body[0].names, tuple) or isinstance(stmt_norm.body[0].names, list)
  assert stmt_norm.body[0].names[0].name.value == "sys"  # type: ignore is forbidden but actually Name isn't complex

  # Exception handling during literal_eval (should fallback to basic import module)
  op_err: OperationNode = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[
      AttributeNode(name="module", value='"sys"'),
      AttributeNode(name="names", value="[invalid"),
    ],
  )
  stmt_err: cst.BaseStatement = gen._convert_import(op_err)
  assert isinstance(stmt_err, cst.SimpleStatementLine)
  assert isinstance(stmt_err.body[0], cst.Import)
  assert isinstance(stmt_err.body[0].names, tuple) or isinstance(stmt_err.body[0].names, list)
  assert getattr(stmt_err.body[0].names[0].name, "value", None) == "sys"

  # Empty names, empty module (should fallback to pass)
  op_empty: OperationNode = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[
      AttributeNode(name="names", value="[]"),
    ],
  )
  stmt_empty: cst.BaseStatement = gen._convert_import(op_empty)
  assert isinstance(stmt_empty, cst.SimpleStatementLine)
  assert isinstance(stmt_empty.body[0], cst.Pass)


def test_convert_return() -> None:
  """Docstring."""
  gen: DummyGenerator = DummyGenerator()

  # Without operands
  op_empty: OperationNode = OperationNode(name="sw.return", operands=[], attributes=[])
  stmt_empty: cst.BaseStatement = gen._convert_return(op_empty)
  assert isinstance(stmt_empty, cst.SimpleStatementLine)
  assert isinstance(stmt_empty.body[0], cst.Return)
  assert stmt_empty.body[0].value is None

  # With operand
  op: OperationNode = OperationNode(name="sw.return", operands=[ValueNode(name="ret_val")], attributes=[])
  stmt: cst.BaseStatement = gen._convert_return(op)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Return)
  assert isinstance(stmt.body[0].value, cst.Name)
  assert stmt.body[0].value.value == "resolved_ret_val"


def test_convert_class_def() -> None:
  """Docstring."""
  gen: DummyGenerator = DummyGenerator()

  # Empty Class
  op_empty: OperationNode = OperationNode(name="sw.module", operands=[], attributes=[])
  cls_empty: cst.ClassDef = gen._convert_class_def(op_empty)
  assert isinstance(cls_empty, cst.ClassDef)
  assert cls_empty.name.value == "UnknownClass"
  assert len(cls_empty.bases) == 0
  assert isinstance(cls_empty.body.body[0], cst.SimpleStatementLine)
  assert isinstance(cls_empty.body.body[0].body[0], cst.Pass)

  # With bases and body
  block: BlockNode = BlockNode(label="block", arguments=[], operations=[])
  region: RegionNode = RegionNode(blocks=[block])
  op: OperationNode = OperationNode(
    name="sw.module",
    operands=[],
    attributes=[
      AttributeNode(name="sym_name", value='"MyClass"'),
      AttributeNode(name="bases", value='["Base1", "module.Base2"]'),
    ],
    regions=[region],
  )
  cls: cst.ClassDef = gen._convert_class_def(op)
  assert cls.name.value == "MyClass"
  assert len(cls.bases) == 2
  assert isinstance(cls.bases[0].value, cst.Name)
  assert cls.bases[0].value.value == "Base1"
  assert isinstance(cls.bases[1].value, cst.Attribute)
  assert isinstance(cls.bases[1].value.value, cst.Name)
  assert cls.bases[1].value.value.value == "module"
  assert cls.bases[1].value.attr.value == "Base2"


def test_convert_func_def() -> None:
  """Docstring."""
  gen: DummyGenerator = DummyGenerator()

  # Empty func
  op_empty: OperationNode = OperationNode(name="sw.func", operands=[], attributes=[])
  func_empty: cst.FunctionDef = gen._convert_func_def(op_empty)
  assert isinstance(func_empty, cst.FunctionDef)
  assert func_empty.name.value == "unknown_func"
  assert len(func_empty.params.params) == 0

  # With args and type hints
  block: BlockNode = BlockNode(
    label="block",
    arguments=[
      (ValueNode(name="arg1"), TypeNode(body='!sw.type<"int">')),
      (ValueNode(name="arg2"), TypeNode(body='!sw.type<"Any">')),
      (ValueNode(name="arg3"), TypeNode(body="!sw.type<'module.Type'>")),
    ],
    operations=[],
  )
  region: RegionNode = RegionNode(blocks=[block])
  op: OperationNode = OperationNode(
    name="sw.func", operands=[], attributes=[AttributeNode(name="sym_name", value='"my_func"')], regions=[region]
  )
  func: cst.FunctionDef = gen._convert_func_def(op)
  assert func.name.value == "my_func"
  params = func.params.params
  assert len(params) == 3

  assert params[0].name.value == "arg"
  assert params[0].annotation is not None
  assert isinstance(params[0].annotation.annotation, cst.Name)
  assert params[0].annotation.annotation.value == "int"

  assert params[1].name.value == "arg2"
  assert params[1].annotation is None  # Any is skipped

  assert params[2].name.value == "arg3"
  assert params[2].annotation is not None
  assert isinstance(params[2].annotation.annotation, cst.Attribute)
  assert isinstance(params[2].annotation.annotation.value, cst.Name)
  assert params[2].annotation.annotation.value.value == "module"
  assert params[2].annotation.annotation.attr.value == "Type"

  # Verify context gets restored
  assert len(gen.ctx._map) == 0


def test_convert_class_def_missing_branches() -> None:
  """Function doc."""
  gen: DummyGenerator = DummyGenerator()
  # clean is empty string
  op_empty_bases: OperationNode = OperationNode(
    name="sw.class",
    attributes=[AttributeNode(name="name", value='"MyClass"'), AttributeNode(name="bases", value="[]")],
    regions=[RegionNode(blocks=[BlockNode(arguments=[], operations=[])])],
  )
  stmt: cst.ClassDef = gen._convert_class_def(op_empty_bases)
  assert len(stmt.bases) == 0

  # b is empty string
  op_empty_b: OperationNode = OperationNode(
    name="sw.class",
    attributes=[AttributeNode(name="name", value='"MyClass"'), AttributeNode(name="bases", value="[Base,,]")],
    regions=[RegionNode(blocks=[BlockNode(arguments=[], operations=[])])],
  )
  stmt_b: cst.ClassDef = gen._convert_class_def(op_empty_b)
  assert len(stmt_b.bases) == 1
  assert isinstance(stmt_b.bases[0].value, cst.Name)
  assert stmt_b.bases[0].value.value == "Base"


def test_convert_func_def_missing_branches() -> None:
  """Function doc."""
  gen: DummyGenerator = DummyGenerator()
  op_no_sw_type: OperationNode = OperationNode(
    name="sw.func",
    attributes=[AttributeNode(name="name", value='"my_func"')],
    regions=[RegionNode(blocks=[BlockNode(arguments=[(ValueNode(name="arg1"), TypeNode(body="i32"))], operations=[])])],
  )
  stmt: cst.FunctionDef = gen._convert_func_def(op_no_sw_type)
  assert len(stmt.params.params) == 1
  assert stmt.params.params[0].annotation is None
