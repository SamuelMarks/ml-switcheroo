"""Tests for MLIR generator coverage."""

import libcst as cst
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.naming import NamingContext
from ml_switcheroo.core.mlir.cst import OperationNode, AttributeNode, BlockNode, ValueNode
from ml_switcheroo.core.cst.base import Trivia


def test_stmt_with_changes_leading_lines():
  """Test stmt with changes leading lines."""
  ctx = NamingContext()
  gen = MlirToPythonGenerator()
  gen.ctx = ctx
  op = OperationNode(
    name="sw.constant",
    operands=[],
    results=[ValueNode(name="%0")],
    attributes=[
      AttributeNode(name="value", value='"42"', type_annotation="str"),
      AttributeNode(name="doc", value='"docstring"', type_annotation="str"),
    ],
    regions=[],
    leading_trivia=[Trivia("comment")],
  )
  block = BlockNode(label="^bb0", operations=[op])
  stmts = gen._convert_block(block)
  # Should hit line 125
  assert len(stmts) == 1


def test_convert_statement_import_and_none():
  """Test convert statement import and none."""
  ctx = NamingContext()
  gen = MlirToPythonGenerator()
  gen.ctx = ctx
  # Give it module name so it creates import correctly
  op1 = OperationNode(
    name="sw.import",
    results=[ValueNode(name="%mod")],
    operands=[],
    attributes=[
      AttributeNode(name="module", value="os", type_annotation="str"),
      AttributeNode(name="names", value="['path']", type_annotation="array"),
      AttributeNode(name="aliases", value="['']", type_annotation="array"),
    ],
    regions=[],
  )
  op2 = OperationNode(name="sw.unknown_stmt", results=[], operands=[], attributes=[], regions=[])
  # _convert_statement_op directly
  assert gen._convert_statement_op(op1) is not None
  assert gen._convert_statement_op(op2) is None


def test_wrap_as_statement_void_call():
  """Test wrap as statement void call."""
  ctx = NamingContext()
  gen = MlirToPythonGenerator()
  gen.ctx = ctx
  op = OperationNode(name="sw.call", operands=[], results=[ValueNode(name="%0")], attributes=[], regions=[])
  gen.usage_counts["%0"] = 1  # Not 0
  expr = cst.parse_expression("super().__init__()")
  stmt = gen._wrap_as_statement(op, expr)
  # Should hit line 247: is_void_call is true for print
  assert isinstance(stmt.body[0], cst.Expr)


def test_wrap_as_statement_getattr():
  """Test wrap as statement getattr."""
  ctx = NamingContext()
  gen = MlirToPythonGenerator()
  gen.ctx = ctx
  op = OperationNode(
    name="sw.getattr",
    operands=[ValueNode(name="%1")],
    results=[ValueNode(name="%0")],
    attributes=[AttributeNode(name="name", value='"myattr"', type_annotation="str")],
    regions=[],
  )
  gen.usage_counts["%0"] = 1
  expr = cst.Name("dummy")
  _ = gen._wrap_as_statement(op, expr)
  # Should hit line 260
  assert "myattr" in ctx._map["%0"]


def test_wrap_as_statement_constant():
  """Test wrap as statement constant."""
  ctx = NamingContext()
  gen = MlirToPythonGenerator()
  gen.ctx = ctx
  op = OperationNode(name="sw.constant", operands=[], results=[ValueNode(name="%0")], attributes=[], regions=[])
  gen.usage_counts["%0"] = 1
  expr = cst.Integer("1")
  _ = gen._wrap_as_statement(op, expr)
  # Should hit line 266
  assert "cst" in ctx._map["%0"]
