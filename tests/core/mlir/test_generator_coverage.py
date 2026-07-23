"""Tests for MLIR generator coverage."""

import libcst as cst
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.naming import NamingContext
from ml_switcheroo.core.mlir.nodes import OperationNode, AttributeNode, BlockNode, ValueNode, TriviaNode


def test_stmt_with_changes_leading_lines():
  """Test stmt with changes leading lines."""
  ctx = NamingContext()
  gen = MlirToPythonGenerator()
  gen.ctx = ctx
  op = OperationNode(
    "sw.constant",
    [],
    [ValueNode(name="%0")],
    [AttributeNode("value", '"42"', "str"), AttributeNode("doc", '"docstring"', "str")],
    [],
    leading_trivia=[TriviaNode(content="comment")],
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
    "sw.import",
    ["%mod"],
    [],
    [
      AttributeNode("module", "os", "str"),
      AttributeNode("names", "['path']", "array"),
      AttributeNode("aliases", "['']", "array"),
    ],
    [],
  )
  op2 = OperationNode("sw.unknown_stmt", [], [], [], [])
  # _convert_statement_op directly
  assert gen._convert_statement_op(op1) is not None
  assert gen._convert_statement_op(op2) is None


def test_wrap_as_statement_void_call():
  """Test wrap as statement void call."""
  ctx = NamingContext()
  gen = MlirToPythonGenerator()
  gen.ctx = ctx
  op = OperationNode("sw.call", operands=[], results=[ValueNode(name="%0")], attributes=[], regions=[])
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
    "sw.getattr",
    operands=["%1"],
    results=[ValueNode(name="%0")],
    attributes=[AttributeNode("name", '"myattr"', "str")],
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
  op = OperationNode("sw.constant", operands=[], results=[ValueNode(name="%0")], attributes=[], regions=[])
  gen.usage_counts["%0"] = 1
  expr = cst.Integer("1")
  _ = gen._wrap_as_statement(op, expr)
  # Should hit line 266
  assert "cst" in ctx._map["%0"]
