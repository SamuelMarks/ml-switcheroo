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
    leading_trivia=[Trivia("// comment")],
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
  expr = cst.parse_expression("super().__init__()")
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


class TriviaWithContent:
  """Trivia object with content attribute."""

  def __init__(self, content: str) -> None:
    """Initialize TriviaWithContent with content.

    Args:
        content: Raw comment content.
    """
    self.content: str = content


class TriviaTextOnly:
  """Trivia object with text attribute instead of content."""

  def __init__(self, text: str) -> None:
    """Initialize TriviaTextOnly with text.

    Args:
        text: Raw comment text.
    """
    self.text: str = text


def test_convert_trivia_text_attribute_and_stmt_leading_lines() -> None:
  """Test trivia conversion using text attribute and statement ops with leading trivia."""
  gen = MlirToPythonGenerator()

  # 1. line 109: Trivia with content attribute and text attribute
  lines1 = gen._convert_trivia([TriviaWithContent("// comment from content")])
  assert len(lines1) == 1
  lines2 = gen._convert_trivia([TriviaTextOnly("// comment from text")])
  assert len(lines2) == 1

  # 2. line 150: Statement op with leading trivia
  op_import = OperationNode(
    name="sw.import",
    results=[ValueNode(name="%mod")],
    operands=[],
    attributes=[
      AttributeNode(name="module", value="math", type_annotation="str"),
      AttributeNode(name="names", value="['sin']", type_annotation="array"),
      AttributeNode(name="aliases", value="['']", type_annotation="array"),
    ],
    regions=[],
    leading_trivia=[Trivia("// leading comment")],
  )
  block = BlockNode(label="^bb0", operations=[op_import])
  stmts = gen._convert_block(block)
  assert len(stmts) == 1


def test_is_void_call_non_super() -> None:
  """Test _is_void_call branches for non-super init calls."""
  gen = MlirToPythonGenerator()

  # 348->351: receiver is Name, not Call
  expr_obj = typing.cast(cst.Call, cst.parse_expression("self.__init__()"))
  assert gen._is_void_call(expr_obj) is False

  # 349->351: receiver is Call, but func is not 'super'
  expr_other = typing.cast(cst.Call, cst.parse_expression("other().__init__()"))
  assert gen._is_void_call(expr_other) is False
