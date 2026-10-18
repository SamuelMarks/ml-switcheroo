"""Test suite for gen statements."""

import typing

from collections import defaultdict

import libcst as _cst
import pytest

from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, OperationNode, RegionNode, TypeNode, ValueNode
from ml_switcheroo.core.mlir.gen_statements import StatementGeneratorMixin
from ml_switcheroo.core.mlir.naming import NamingContext

cst: typing.Any = _cst


class DummyGenerator(StatementGeneratorMixin):
  """Test helper class DummyGenerator."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.ctx: typing.Any = NamingContext()
    self.usage_counts: typing.Any = defaultdict(int)
    self.usage_consumers: typing.Any = {}
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

  def _convert_block(self, block: typing.Any) -> typing.Any:
    """Execute convert block helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return [cst.SimpleStatementLine(body=[cst.Pass()])]

  def _scan_block_usage(self, block: typing.Any) -> None:
    """Execute scan block usage helper logic."""
    pass


def test_resolve_operand_not_implemented() -> None:
  """Verify resolve operand not implemented behavior."""

  class IncompleteGenerator(StatementGeneratorMixin):
    """Test helper class IncompleteGenerator."""

    def _convert_block(self, block: typing.Any) -> typing.Any:
      """Execute convert block helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return []

    def _scan_block_usage(self, block: typing.Any) -> None:
      """Execute scan block usage helper logic."""
      pass

  gen: typing.Any = IncompleteGenerator()
  with pytest.raises(NotImplementedError):
    gen._resolve_operand("foo")


def test_convert_block_not_implemented() -> None:
  """Verify convert block not implemented behavior."""

  class IncompleteGenerator(StatementGeneratorMixin):
    """Test helper class IncompleteGenerator."""

    def _resolve_operand(self, ssa_name: typing.Any) -> typing.Any:
      """Execute resolve operand helper logic.

      Args:
          ssa_name: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return cst.Name("dummy")

    def _scan_block_usage(self, block: typing.Any) -> None:
      """Execute scan block usage helper logic."""
      pass

  gen: typing.Any = IncompleteGenerator()
  with pytest.raises(NotImplementedError):
    gen._convert_block(BlockNode(label="foo"))


def test_scan_block_usage_not_implemented() -> None:
  """Verify scan block usage not implemented behavior."""

  class IncompleteGenerator(StatementGeneratorMixin):
    """Test helper class IncompleteGenerator."""

    def _resolve_operand(self, ssa_name: typing.Any) -> typing.Any:
      """Execute resolve operand helper logic.

      Args:
          ssa_name: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return cst.Name("dummy")

    def _convert_block(self, block: typing.Any) -> typing.Any:
      """Execute convert block helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return []

  gen: typing.Any = IncompleteGenerator()
  with pytest.raises(NotImplementedError):
    gen._scan_block_usage(BlockNode(label="foo"))


def test_convert_setattr() -> None:
  """Verify convert setattr behavior."""
  gen: typing.Any = DummyGenerator()

  # Missing operands
  op_err: typing.Any = OperationNode(name="sw.setattr", operands=[], attributes=[])
  stmt_err: typing.Any = gen._convert_setattr(op_err)
  assert isinstance(stmt_err, cst.SimpleStatementLine)
  assert isinstance(stmt_err.body[0], cst.Pass)

  # Valid
  op: typing.Any = OperationNode(
    name="sw.setattr",
    operands=[ValueNode(name="obj"), ValueNode(name="val")],
    attributes=[AttributeNode(name="name", value='"my_attr"')],
  )
  stmt: typing.Any = gen._convert_setattr(op)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assign: typing.Any = stmt.body[0]
  assert isinstance(assign, cst.Assign)
  assert isinstance(assign.targets[0].target, cst.Attribute)
  assert isinstance(assign.targets[0].target.value, cst.Name)
  assert assign.targets[0].target.value.value == "resolved_obj"
  assert assign.targets[0].target.attr.value == "my_attr"
  assert isinstance(assign.value, cst.Name)
  assert assign.value.value == "resolved_val"


def test_convert_import() -> None:
  """Verify convert import behavior."""
  gen: typing.Any = DummyGenerator()

  # From import multiple names and aliases
  op_from: typing.Any = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[
      AttributeNode(name="module", value='"os.path"'),
      AttributeNode(name="names", value="['join', 'exists']"),
      AttributeNode(name="aliases", value="['join', 'path_exists']"),
    ],
  )
  stmt: typing.Any = gen._convert_import(op_from)
  assert isinstance(stmt, cst.SimpleStatementLine)
  import_node: typing.Any = stmt.body[0]
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
  op_star: typing.Any = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[
      AttributeNode(name="module", value='"os"'),
      AttributeNode(name="names", value="['*']"),
      AttributeNode(name="aliases", value="['*']"),
    ],
  )
  stmt_star: typing.Any = gen._convert_import(op_star)
  assert isinstance(stmt_star, cst.SimpleStatementLine)
  assert isinstance(stmt_star.body[0], cst.ImportFrom)
  assert isinstance(stmt_star.body[0].names, cst.ImportStar)

  # Import normal
  op_norm: typing.Any = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[AttributeNode(name="names", value="['sys']"), AttributeNode(name="aliases", value="['sys']")],
  )
  stmt_norm: typing.Any = gen._convert_import(op_norm)
  assert isinstance(stmt_norm, cst.SimpleStatementLine)
  assert isinstance(stmt_norm.body[0], cst.Import)
  assert isinstance(stmt_norm.body[0].names, tuple) or isinstance(stmt_norm.body[0].names, list)
  assert stmt_norm.body[0].names[0].name.value == "sys"

  # Exception handling during literal_eval (should fallback to basic import module)
  op_err: typing.Any = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[
      AttributeNode(name="module", value='"sys"'),
      AttributeNode(name="names", value="[invalid"),
    ],
  )
  stmt_err: typing.Any = gen._convert_import(op_err)
  assert isinstance(stmt_err, cst.SimpleStatementLine)
  assert isinstance(stmt_err.body[0], cst.Import)
  assert isinstance(stmt_err.body[0].names, tuple) or isinstance(stmt_err.body[0].names, list)
  assert getattr(stmt_err.body[0].names[0].name, "value", None) == "sys"

  # Empty names, empty module (should fallback to pass)
  op_empty: typing.Any = OperationNode(
    name="sw.import",
    operands=[],
    attributes=[
      AttributeNode(name="names", value="[]"),
    ],
  )
  stmt_empty: typing.Any = gen._convert_import(op_empty)
  assert isinstance(stmt_empty, cst.SimpleStatementLine)
  assert isinstance(stmt_empty.body[0], cst.Pass)


def test_convert_return() -> None:
  """Verify convert return behavior."""
  gen: typing.Any = DummyGenerator()

  # Without operands
  op_empty: typing.Any = OperationNode(name="sw.return", operands=[], attributes=[])
  stmt_empty: typing.Any = gen._convert_return(op_empty)
  assert isinstance(stmt_empty, cst.SimpleStatementLine)
  assert isinstance(stmt_empty.body[0], cst.Return)
  assert stmt_empty.body[0].value is None

  # With operand
  op: typing.Any = OperationNode(name="sw.return", operands=[ValueNode(name="ret_val")], attributes=[])
  stmt: typing.Any = gen._convert_return(op)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Return)
  assert isinstance(stmt.body[0].value, cst.Name)
  assert stmt.body[0].value.value == "resolved_ret_val"


def test_convert_class_def() -> None:
  """Verify convert class def behavior."""
  gen: typing.Any = DummyGenerator()

  # Empty Class
  op_empty: typing.Any = OperationNode(name="sw.module", operands=[], attributes=[])
  cls_empty: typing.Any = gen._convert_class_def(op_empty)
  assert isinstance(cls_empty, cst.ClassDef)
  assert cls_empty.name.value == "UnknownClass"
  assert len(cls_empty.bases) == 0
  assert isinstance(cls_empty.body.body[0], cst.SimpleStatementLine)
  assert isinstance(cls_empty.body.body[0].body[0], cst.Pass)

  # With bases and body
  block: typing.Any = BlockNode(label="block", arguments=[], operations=[])
  region: typing.Any = RegionNode(blocks=[block])
  op: typing.Any = OperationNode(
    name="sw.module",
    operands=[],
    attributes=[
      AttributeNode(name="sym_name", value='"MyClass"'),
      AttributeNode(name="bases", value='["Base1", "module.Base2"]'),
    ],
    regions=[region],
  )
  cls: typing.Any = gen._convert_class_def(op)
  assert cls.name.value == "MyClass"
  assert len(cls.bases) == 2
  assert isinstance(cls.bases[0].value, cst.Name)
  assert cls.bases[0].value.value == "Base1"
  assert isinstance(cls.bases[1].value, cst.Attribute)
  assert isinstance(cls.bases[1].value.value, cst.Name)
  assert cls.bases[1].value.value.value == "module"
  assert cls.bases[1].value.attr.value == "Base2"


def test_convert_func_def() -> None:
  """Verify convert func def behavior."""
  gen: typing.Any = DummyGenerator()

  # Empty func
  op_empty: typing.Any = OperationNode(name="sw.func", operands=[], attributes=[])
  func_empty: typing.Any = gen._convert_func_def(op_empty)
  assert isinstance(func_empty, cst.FunctionDef)
  assert func_empty.name.value == "unknown_func"
  assert len(func_empty.params.params) == 0

  # With args and type hints
  block: typing.Any = BlockNode(
    label="block",
    arguments=[
      (ValueNode(name="arg1"), TypeNode(body='!sw.type<"int">')),
      (ValueNode(name="arg2"), TypeNode(body='!sw.type<"Any">')),
      (ValueNode(name="arg3"), TypeNode(body="!sw.type<'module.Type'>")),
    ],
    operations=[],
  )
  region: typing.Any = RegionNode(blocks=[block])
  op: typing.Any = OperationNode(
    name="sw.func", operands=[], attributes=[AttributeNode(name="sym_name", value='"my_func"')], regions=[region]
  )
  func: typing.Any = gen._convert_func_def(op)
  assert func.name.value == "my_func"
  params: typing.Any = func.params.params
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
  """Verify convert class def missing branches behavior."""
  gen: typing.Any = DummyGenerator()
  # clean is empty string
  op_empty_bases: typing.Any = OperationNode(
    name="sw.class",
    attributes=[AttributeNode(name="name", value='"MyClass"'), AttributeNode(name="bases", value="[]")],
    regions=[RegionNode(blocks=[BlockNode(arguments=[], operations=[])])],
  )
  stmt: typing.Any = gen._convert_class_def(op_empty_bases)
  assert len(stmt.bases) == 0

  # b is empty string
  op_empty_b: typing.Any = OperationNode(
    name="sw.class",
    attributes=[AttributeNode(name="name", value='"MyClass"'), AttributeNode(name="bases", value="[Base,,]")],
    regions=[RegionNode(blocks=[BlockNode(arguments=[], operations=[])])],
  )
  stmt_b: typing.Any = gen._convert_class_def(op_empty_b)
  assert len(stmt_b.bases) == 1
  assert isinstance(stmt_b.bases[0].value, cst.Name)
  assert stmt_b.bases[0].value.value == "Base"


def test_convert_func_def_missing_branches() -> None:
  """Verify convert func def missing branches behavior."""
  gen: typing.Any = DummyGenerator()
  op_no_sw_type: typing.Any = OperationNode(
    name="sw.func",
    attributes=[AttributeNode(name="name", value='"my_func"')],
    regions=[RegionNode(blocks=[BlockNode(arguments=[(ValueNode(name="arg1"), TypeNode(body="i32"))], operations=[])])],
  )
  stmt: typing.Any = gen._convert_func_def(op_no_sw_type)
  assert len(stmt.params.params) == 1
  assert stmt.params.params[0].annotation is None
