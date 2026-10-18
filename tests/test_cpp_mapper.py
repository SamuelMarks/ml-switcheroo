"""Test suite for cpp mapper."""

import typing

import ast
import os
from unittest.mock import mock_open, patch

import pytest

from ml_switcheroo.core.compiler.backends.cpp.cst import BinaryExpression, Identifier, MethodCall
from ml_switcheroo.core.compiler.backends.cpp.mapper import ASTToCppMapper


def test_mapper_init_no_file(monkeypatch: typing.Any) -> None:
  """Verify mapper init no file behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  monkeypatch.setattr(os.path, "exists", lambda p: False)
  mapper: typing.Any = ASTToCppMapper()
  assert mapper.op_map == {}


def test_mapper_init_with_file() -> None:
  """Verify mapper init with file behavior."""
  mock_data: typing.Any = '{"Add": "+"}'
  with patch("os.path.exists", return_value=True), patch("builtins.open", mock_open(read_data=mock_data)):
    mapper: typing.Any = ASTToCppMapper()
    assert mapper.op_map == {"Add": "+"}


def test_mapper_expression_name() -> None:
  """Verify mapper expression name behavior."""
  mapper: typing.Any = ASTToCppMapper()
  node: typing.Any = ast.Name(id="foo")
  expr: typing.Any = mapper.map_expression(node)
  assert isinstance(expr, Identifier)
  assert expr.name == "foo"


def test_mapper_expression_binop() -> None:
  """Verify mapper expression binop behavior."""
  mapper: typing.Any = ASTToCppMapper()
  mapper.op_map = {"Add": "+"}
  node: typing.Any = ast.BinOp(left=ast.Name(id="a"), op=ast.Add(), right=ast.Name(id="b"))
  expr: typing.Any = mapper.map_expression(node)
  assert isinstance(expr, BinaryExpression)
  assert expr.operator == "+"
  assert isinstance(expr.left, Identifier)
  assert isinstance(expr.right, Identifier)
  assert expr.left.name == "a"
  assert expr.right.name == "b"


def test_mapper_expression_binop_unknown_op() -> None:
  """Verify mapper expression binop unknown op behavior."""
  mapper: typing.Any = ASTToCppMapper()
  mapper.op_map = {}
  node: typing.Any = ast.BinOp(left=ast.Name(id="a"), op=ast.Sub(), right=ast.Name(id="b"))
  expr: typing.Any = mapper.map_expression(node)
  assert isinstance(expr, BinaryExpression)
  assert expr.operator == "+"  # Default


def test_mapper_expression_call_name() -> None:
  """Verify mapper expression call name behavior."""
  mapper: typing.Any = ASTToCppMapper()
  node: typing.Any = ast.Call(func=ast.Name(id="foo"), args=[ast.Name(id="a")], keywords=[])
  expr: typing.Any = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "foo"
  assert len(expr.arguments) == 1
  assert isinstance(expr.arguments[0], Identifier)
  assert expr.arguments[0].name == "a"


def test_mapper_expression_call_attribute() -> None:
  """Verify mapper expression call attribute behavior."""
  mapper: typing.Any = ASTToCppMapper()
  node: typing.Any = ast.Call(func=ast.Attribute(value=ast.Name(id="obj"), attr="method"), args=[], keywords=[])
  expr: typing.Any = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "obj.method"


def test_mapper_expression_call_unknown() -> None:
  """Verify mapper expression call unknown behavior."""
  mapper: typing.Any = ASTToCppMapper()
  # Call with a function that is not Name or Attribute (e.g. Call of Call)
  node: typing.Any = ast.Call(func=ast.Call(func=ast.Name(id="f"), args=[], keywords=[]), args=[], keywords=[])
  expr: typing.Any = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "unknown"


def test_mapper_expression_call_attribute_complex() -> None:
  """Verify mapper expression call attribute complex behavior."""
  mapper: typing.Any = ASTToCppMapper()
  # Call with an attribute where value is not a Name
  node: typing.Any = ast.Call(
    func=ast.Attribute(value=ast.Call(func=ast.Name(id="f"), args=[], keywords=[]), attr="method"), args=[], keywords=[]
  )
  expr: typing.Any = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "method"


def test_mapper_expression_constant() -> None:
  """Verify mapper expression constant behavior."""
  mapper: typing.Any = ASTToCppMapper()
  node: typing.Any = ast.Constant(value=42)
  expr: typing.Any = mapper.map_expression(node)
  assert isinstance(expr, Identifier)
  assert expr.name == "42"


def test_mapper_expression_unsupported() -> None:
  """Verify mapper expression unsupported behavior."""
  mapper: typing.Any = ASTToCppMapper()
  node: typing.Any = ast.List(elts=[])
  with pytest.raises(ValueError):
    mapper.map_expression(node)
