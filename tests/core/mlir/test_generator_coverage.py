"""Tests for MLIR generator coverage."""

import typing

import libcst as cst

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, OperationNode, ValueNode
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.naming import NamingContext


def test_stmt_with_changes_leading_lines() -> None:
  """Docstring."""
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
  stmts: list[typing.Any] = gen._convert_block(block)
  # Should hit line 125
  assert len(stmts) == 1


def test_convert_statement_import_and_none() -> None:
  """Docstring."""
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


def test_wrap_as_statement_void_call() -> None:
  """Docstring."""
  ctx = NamingContext()
  gen = MlirToPythonGenerator()
  gen.ctx = ctx
  op = OperationNode(name="sw.call", operands=[], results=[ValueNode(name="%0")], attributes=[], regions=[])
  gen.usage_counts["%0"] = 1  # Not 0
  expr = typing.cast(cst.Expr, cst.parse_expression("super().__init__()"))
  stmt: typing.Any = gen._wrap_as_statement(op, expr)
  # Should hit line 247: is_void_call is true for print
  assert isinstance(stmt.body[0], cst.Expr)


def test_wrap_as_statement_getattr() -> None:
  """Docstring."""
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


def test_wrap_as_statement_constant() -> None:
  """Docstring."""
  ctx = NamingContext()
  gen = MlirToPythonGenerator()
  gen.ctx = ctx
  op = OperationNode(name="sw.constant", operands=[], results=[ValueNode(name="%0")], attributes=[], regions=[])
  gen.usage_counts["%0"] = 1
  expr = cst.Integer("1")
  _ = gen._wrap_as_statement(op, expr)
  # Should hit line 266
  assert "cst" in ctx._map["%0"]
