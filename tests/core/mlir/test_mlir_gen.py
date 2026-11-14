"""Unit tests for ``MlirExpressionGenerator`` and ``MlirStatementGenerator`` delegates."""

from collections import defaultdict

import libcst as cst

from ml_switcheroo.core.mlir.cst import (
  AttributeNode,
  BlockNode,
  OperationNode,
  RegionNode,
  Trivia,
  TypeNode,
  ValueNode,
)
from ml_switcheroo.core.mlir.gen_base import MlirGenHelpers
from ml_switcheroo.core.mlir.gen_expressions import MlirExpressionGenerator
from ml_switcheroo.core.mlir.gen_statements import MlirStatementGenerator
from ml_switcheroo.core.mlir.naming import NamingContext


class StubGeneratorHost:
  """Typed host implementing ``MlirExprGeneratorHost`` and ``MlirStmtGeneratorHost``."""

  def __init__(self) -> None:
    """Initialize stub host state and delegates."""
    self.ctx: NamingContext = NamingContext()
    self.usage_counts: defaultdict[str, int] = defaultdict(int)
    self.usage_consumers: dict[str, OperationNode] = {}
    self.resolved_values: dict[str, cst.BaseExpression] = {}
    self.helpers: MlirGenHelpers = MlirGenHelpers()
    self.expr_gen: MlirExpressionGenerator = MlirExpressionGenerator(self, self.helpers)
    self.stmt_gen: MlirStatementGenerator = MlirStatementGenerator(self, self.helpers)

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Resolve ``ssa_name`` to a mapped or synthetic ``cst.Name``.

    Args:
        ssa_name: SSA value identifier string.

    Returns:
        Resolved ``cst.BaseExpression``.
    """
    if ssa_name in self.resolved_values:
      return self.resolved_values[ssa_name]
    return cst.Name(f"res_{ssa_name.replace('%', '')}")

  def _convert_block(self, block: BlockNode) -> list[cst.SimpleStatementLine | cst.BaseCompoundStatement]:
    """Convert ``block`` into empty list when empty or a ``pass`` line otherwise.

    Args:
        block: MLIR ``BlockNode`` to convert.

    Returns:
        List of LibCST statements.
    """
    if not block.operations:
      return []
    return [cst.SimpleStatementLine(body=[cst.Pass()])]

  def _scan_block_usage(self, block: BlockNode) -> None:
    """No-op usage scan stub for isolated statement tests.

    Args:
        block: MLIR ``BlockNode`` being scanned.
    """


def test_expression_generator_keywords_constants_getattr_and_calls() -> None:
  """Verify ``MlirExpressionGenerator`` keyword parsing, constants, getattr, calls, and binops."""
  host = StubGeneratorHost()
  expr_gen = MlirExpressionGenerator(host)

  op_str = OperationNode(name="sw.call", attributes=[AttributeNode(name="arg_keywords", value='["arg1", "arg2"]')])
  assert expr_gen.parse_keywords(op_str) == ["arg1", "arg2"]

  op_list = OperationNode(name="sw.op", attributes=[AttributeNode(name="arg_keywords", value=['"k1"', '"k2"'])])
  assert expr_gen.parse_keywords(op_list) == ["k1", "k2"]

  op_invalid = OperationNode(name="sw.call", attributes=[AttributeNode(name="arg_keywords", value="[invalid")])
  assert expr_gen.parse_keywords(op_invalid) == []

  op_non_list_str = OperationNode(name="sw.call", attributes=[AttributeNode(name="arg_keywords", value='"not a list"')])
  assert expr_gen.parse_keywords(op_non_list_str) == []

  op_non_str = OperationNode(
    name="sw.op",
    attributes=[AttributeNode(name="arg_keywords", value=123), AttributeNode(name="other", value="ignored")],
  )
  assert expr_gen.parse_keywords(op_non_str) == []

  res_bad_const = expr_gen.expr_sw_constant(
    OperationNode(name="sw.constant", attributes=[AttributeNode(name="value", value="yield")])
  )
  assert isinstance(res_bad_const, cst.Name)

  assert isinstance(expr_gen.expr_sw_getattr(OperationNode(name="sw.getattr")), cst.Name)
  assert isinstance(
    expr_gen.expr_sw_getattr(
      OperationNode(
        name="sw.getattr",
        operands=[ValueNode(name="%obj")],
        attributes=[AttributeNode(name="name", value='"attr"')],
      )
    ),
    cst.Attribute,
  )

  res_empty_call = expr_gen.expr_sw_call(OperationNode(name="sw.call"))
  assert isinstance(res_empty_call, cst.Call)

  res_kw_call = expr_gen.expr_sw_call(
    OperationNode(
      name="sw.call",
      operands=[ValueNode(name="%func"), ValueNode(name="%a1"), ValueNode(name="%a2")],
      attributes=[AttributeNode(name="arg_keywords", value='["", "kw1"]')],
    )
  )
  assert isinstance(res_kw_call, cst.Call)
  assert res_kw_call.args[0].keyword is None
  assert res_kw_call.args[1].keyword is not None
  assert res_kw_call.args[1].keyword.value == "kw1"

  res_op = expr_gen.expr_sw_op(
    OperationNode(
      name="sw.op",
      operands=[ValueNode(name="%x"), ValueNode(name="%y")],
      attributes=[AttributeNode(name="type", value='"torch.add"'), AttributeNode(name="arg_keywords", value='["", "k"]')],
    )
  )
  assert isinstance(res_op, cst.Call)
  assert res_op.args[0].keyword is None
  assert res_op.args[1].keyword is not None

  assert isinstance(
    expr_gen.expr_binop(OperationNode(name="sw.op", operands=[ValueNode(name="%1")]), "binop.add"),
    cst.Name,
  )
  for op_code, expected_cls in [
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
    ("unknown_op", cst.Add),
  ]:
    bin_op = OperationNode(
      name="sw.op",
      operands=[ValueNode(name="%1"), ValueNode(name="%2")],
      attributes=[AttributeNode(name="type", value=f'"binop.{op_code}"')],
    )
    res_bin = expr_gen.expr_sw_op(bin_op)
    assert isinstance(res_bin, cst.BinaryOperation)
    assert isinstance(res_bin.operator, expected_cls)


def test_statement_generator_setattr_import_return_class_and_func() -> None:
  """Verify ``MlirStatementGenerator`` across setattr, import, return, class, and function ops."""
  host = StubGeneratorHost()
  stmt_gen = MlirStatementGenerator(host)

  assert isinstance(stmt_gen.convert_setattr(OperationNode(name="sw.setattr")).body[0], cst.Pass)
  assert isinstance(
    stmt_gen.convert_setattr(
      OperationNode(
        name="sw.setattr",
        operands=[ValueNode(name="%obj"), ValueNode(name="%val")],
        attributes=[AttributeNode(name="name", value='"attr"')],
      )
    ).body[0],
    cst.Assign,
  )

  op_star = OperationNode(
    name="sw.import",
    attributes=[
      AttributeNode(name="module", value='"math"'),
      AttributeNode(name="names", value='["*"]'),
      AttributeNode(name="aliases", value='[""]'),
    ],
  )
  assert isinstance(stmt_gen.convert_import(op_star).body[0], cst.ImportFrom)

  op_non_list = OperationNode(
    name="sw.import",
    attributes=[
      AttributeNode(name="names", value="123"),
      AttributeNode(name="aliases", value="456"),
    ],
  )
  assert isinstance(stmt_gen.convert_import(op_non_list).body[0], cst.Pass)

  op_mod_only = OperationNode(name="sw.import", attributes=[AttributeNode(name="module", value='"os"')])
  assert isinstance(stmt_gen.convert_import(op_mod_only).body[0], cst.Import)

  ret_none = stmt_gen.convert_return(OperationNode(name="sw.return")).body[0]
  assert isinstance(ret_none, cst.Return)
  assert ret_none.value is None
  ret_with_val = stmt_gen.convert_return(OperationNode(name="sw.return", operands=[ValueNode(name="%val")])).body[0]
  assert isinstance(ret_with_val, cst.Return)
  assert ret_with_val.value is not None

  cls_with_body = stmt_gen.convert_class_def(
    OperationNode(
      name="sw.module",
      attributes=[AttributeNode(name="sym_name", value='"MyClass"'), AttributeNode(name="bases", value="[, ,]")],
      regions=[RegionNode(blocks=[BlockNode(label="^bb0", operations=[OperationNode(name="sw.pass")])])],
    )
  )
  assert cls_with_body.name.value == "MyClass"
  assert len(cls_with_body.bases) == 0

  func_from_trivia = stmt_gen.convert_func_def(
    OperationNode(
      name="func.func",
      name_trivia=[Trivia(" "), Trivia("@from_trivia")],
      regions=[
        RegionNode(
          blocks=[
            BlockNode(
              label="^bb0",
              arguments=[
                (ValueNode(name="%a0"), TypeNode(body='!sw.type<"torch.Tensor">')),
                (ValueNode(name="%a1"), TypeNode(body='!sw.type<"Any">')),
                (ValueNode(name="%a2"), TypeNode(body="tensor<f32>")),
              ],
              operations=[OperationNode(name="sw.pass")],
            )
          ]
        )
      ],
    )
  )
  assert func_from_trivia.name.value == "from_trivia"
  assert len(func_from_trivia.params.params) == 3
  assert func_from_trivia.params.params[0].annotation is not None
  assert func_from_trivia.params.params[1].annotation is None
  assert func_from_trivia.params.params[2].annotation is None

  func_blank_trivia = stmt_gen.convert_func_def(OperationNode(name="func.func", name_trivia=[Trivia(" ")]))
  assert func_blank_trivia.name.value == "unknown_func"
