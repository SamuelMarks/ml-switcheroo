"""Test suite for ``MlirStatementGenerator``."""

from collections import defaultdict

import libcst as cst

from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, OperationNode, RegionNode, TypeNode, ValueNode
from ml_switcheroo.core.mlir.gen_statements import MlirStatementGenerator
from ml_switcheroo.core.mlir.naming import NamingContext


class DummyStmtHost:
  """Typed host implementing ``MlirStmtGeneratorHost`` for unit tests."""

  def __init__(self) -> None:
    """Initialize naming and SSA usage tracking structures."""
    self.ctx: NamingContext = NamingContext()
    self.usage_counts: defaultdict[str, int] = defaultdict(int)
    self.usage_consumers: dict[str, OperationNode] = {}
    self._resolved: dict[str, cst.BaseExpression] = {}

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Resolve ``ssa_name`` to its mapped expression or ``resolved_<ssa_name>``.

    Args:
        ssa_name: SSA identifier string.

    Returns:
        Resolved ``cst.BaseExpression``.
    """
    if ssa_name in self._resolved:
      return self._resolved[ssa_name]
    return cst.Name(f"resolved_{ssa_name}")

  def _convert_block(self, block: BlockNode) -> list[cst.SimpleStatementLine | cst.BaseCompoundStatement]:
    """Return a single ``pass`` statement line for testing.

    Args:
        block: MLIR ``BlockNode`` to convert.

    Returns:
        Single-element statement list containing ``pass``.
    """
    return [cst.SimpleStatementLine(body=[cst.Pass()])]

  def _scan_block_usage(self, block: BlockNode) -> None:
    """No-op usage scan stub for isolated statement tests.

    Args:
        block: MLIR ``BlockNode`` being scanned.
    """


def test_convert_setattr_and_return() -> None:
  """Verify ``convert_setattr`` and ``convert_return`` handle missing and valid operands."""
  gen = MlirStatementGenerator(DummyStmtHost())

  stmt_err = gen.convert_setattr(OperationNode(name="sw.setattr"))
  assert isinstance(stmt_err.body[0], cst.Pass)

  stmt_ok = gen.convert_setattr(
    OperationNode(
      name="sw.setattr",
      operands=[ValueNode(name="obj"), ValueNode(name="val")],
      attributes=[AttributeNode(name="name", value='"my_attr"')],
    )
  )
  assign = stmt_ok.body[0]
  assert isinstance(assign, cst.Assign)
  assert isinstance(assign.targets[0].target, cst.Attribute)
  assert assign.targets[0].target.attr.value == "my_attr"

  ret_empty = gen.convert_return(OperationNode(name="sw.return")).body[0]
  assert isinstance(ret_empty, cst.Return)
  assert ret_empty.value is None

  ret_val = gen.convert_return(OperationNode(name="sw.return", operands=[ValueNode(name="ret_val")])).body[0]
  assert isinstance(ret_val, cst.Return)
  assert isinstance(ret_val.value, cst.Name)
  assert ret_val.value.value == "resolved_ret_val"


def test_convert_import_variants() -> None:
  """Verify ``convert_import`` reconstructs ``from ... import``, star imports, plain imports, and fallbacks."""
  gen = MlirStatementGenerator(DummyStmtHost())

  op_from = OperationNode(
    name="sw.import",
    attributes=[
      AttributeNode(name="module", value='"os.path"'),
      AttributeNode(name="names", value="['join', 'exists']"),
      AttributeNode(name="aliases", value="['join', 'path_exists']"),
    ],
  )
  imp_from = gen.convert_import(op_from).body[0]
  assert isinstance(imp_from, cst.ImportFrom)
  assert not isinstance(imp_from.names, cst.ImportStar)
  assert len(imp_from.names) == 2
  assert imp_from.names[0].asname is None
  assert imp_from.names[1].asname is not None
  assert isinstance(imp_from.names[1].asname.name, cst.Name)
  assert imp_from.names[1].asname.name.value == "path_exists"

  op_star = OperationNode(
    name="sw.import",
    attributes=[
      AttributeNode(name="module", value='"os"'),
      AttributeNode(name="names", value="['*']"),
      AttributeNode(name="aliases", value="['*']"),
    ],
  )
  imp_star = gen.convert_import(op_star).body[0]
  assert isinstance(imp_star, cst.ImportFrom)
  assert isinstance(imp_star.names, cst.ImportStar)

  op_norm = OperationNode(
    name="sw.import",
    attributes=[AttributeNode(name="names", value="['sys']"), AttributeNode(name="aliases", value="['sys']")],
  )
  imp_norm = gen.convert_import(op_norm).body[0]
  assert isinstance(imp_norm, cst.Import)

  op_err = OperationNode(
    name="sw.import",
    attributes=[AttributeNode(name="module", value='"sys"'), AttributeNode(name="names", value="[invalid")],
  )
  assert isinstance(gen.convert_import(op_err).body[0], cst.Import)

  op_empty = OperationNode(name="sw.import", attributes=[AttributeNode(name="names", value="[]")])
  assert isinstance(gen.convert_import(op_empty).body[0], cst.Pass)


def test_convert_class_and_func_defs() -> None:
  """Verify ``convert_class_def`` and ``convert_func_def`` reconstruct bases, parameters, and annotations."""
  host = DummyStmtHost()
  gen = MlirStatementGenerator(host)

  cls_empty = gen.convert_class_def(OperationNode(name="sw.module"))
  assert cls_empty.name.value == "UnknownClass"
  assert len(cls_empty.bases) == 0

  cls_empty_brackets = gen.convert_class_def(
    OperationNode(name="sw.module", attributes=[AttributeNode(name="bases", value="[]")])
  )
  assert len(cls_empty_brackets.bases) == 0

  cls_bases = gen.convert_class_def(
    OperationNode(
      name="sw.module",
      attributes=[
        AttributeNode(name="sym_name", value='"MyClass"'),
        AttributeNode(name="bases", value='["Base1", "module.Base2", ""]'),
      ],
      regions=[RegionNode(blocks=[BlockNode(label="block")])],
    )
  )
  assert cls_bases.name.value == "MyClass"
  assert len(cls_bases.bases) == 2

  func_empty = gen.convert_func_def(OperationNode(name="sw.func"))
  assert func_empty.name.value == "unknown_func"

  block = BlockNode(
    label="block",
    arguments=[
      (ValueNode(name="arg1"), TypeNode(body='!sw.type<"int">')),
      (ValueNode(name="arg2"), TypeNode(body='!sw.type<"Any">')),
      (ValueNode(name="arg3"), TypeNode(body="!sw.type<'module.Type'>")),
      (ValueNode(name="arg4"), TypeNode(body="i32")),
    ],
  )
  func = gen.convert_func_def(
    OperationNode(
      name="sw.func",
      attributes=[AttributeNode(name="sym_name", value='"my_func"')],
      regions=[RegionNode(blocks=[block])],
    )
  )
  assert func.name.value == "my_func"
  assert len(func.params.params) == 4
  assert func.params.params[0].annotation is not None
  assert func.params.params[1].annotation is None
  assert func.params.params[2].annotation is not None
  assert func.params.params[3].annotation is None
  assert len(host.ctx._map) == 0
