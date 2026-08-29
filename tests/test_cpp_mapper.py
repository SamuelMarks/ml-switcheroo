"""Test module."""

import ast
import os
from unittest.mock import mock_open, patch

import pytest

from ml_switcheroo.core.compiler.backends.cpp.cst import BinaryExpression, CppNode, Identifier, MethodCall
from ml_switcheroo.core.compiler.backends.cpp.mapper import ASTToCppMapper


def test_mapper_init_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  monkeypatch.setattr(os.path, "exists", lambda p: False)
  mapper: ASTToCppMapper = ASTToCppMapper()
  assert mapper.op_map == {}


def test_mapper_init_with_file() -> None:
  """Docstring."""
  mock_data: str = '{"Add": "+"}'
  with patch("os.path.exists", return_value=True), patch("builtins.open", mock_open(read_data=mock_data)):
    mapper: ASTToCppMapper = ASTToCppMapper()
    assert mapper.op_map == {"Add": "+"}


def test_mapper_expression_name() -> None:
  """Docstring."""
  mapper: ASTToCppMapper = ASTToCppMapper()
  node: ast.Name = ast.Name(id="foo")
  expr: CppNode = mapper.map_expression(node)
  assert isinstance(expr, Identifier)
  assert expr.name == "foo"


def test_mapper_expression_binop() -> None:
  """Docstring."""
  mapper: ASTToCppMapper = ASTToCppMapper()
  mapper.op_map = {"Add": "+"}
  node: ast.BinOp = ast.BinOp(left=ast.Name(id="a"), op=ast.Add(), right=ast.Name(id="b"))
  expr: CppNode = mapper.map_expression(node)
  assert isinstance(expr, BinaryExpression)
  assert expr.operator == "+"
  assert expr.left.name == "a"
  assert expr.right.name == "b"


def test_mapper_expression_binop_unknown_op() -> None:
  """Docstring."""
  mapper: ASTToCppMapper = ASTToCppMapper()
  mapper.op_map = {}
  node: ast.BinOp = ast.BinOp(left=ast.Name(id="a"), op=ast.Sub(), right=ast.Name(id="b"))
  expr: CppNode = mapper.map_expression(node)
  assert isinstance(expr, BinaryExpression)
  assert expr.operator == "+"  # Default


def test_mapper_expression_call_name() -> None:
  """Docstring."""
  mapper: ASTToCppMapper = ASTToCppMapper()
  node: ast.Call = ast.Call(func=ast.Name(id="foo"), args=[ast.Name(id="a")], keywords=[])
  expr: CppNode = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "foo"
  assert len(expr.arguments) == 1
  assert expr.arguments[0].name == "a"


def test_mapper_expression_call_attribute() -> None:
  """Docstring."""
  mapper: ASTToCppMapper = ASTToCppMapper()
  node: ast.Call = ast.Call(func=ast.Attribute(value=ast.Name(id="obj"), attr="method"), args=[], keywords=[])
  expr: CppNode = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "obj.method"


def test_mapper_expression_call_unknown() -> None:
  """Docstring."""
  mapper: ASTToCppMapper = ASTToCppMapper()
  # Call with a function that is not Name or Attribute (e.g. Call of Call)
  node: ast.Call = ast.Call(func=ast.Call(func=ast.Name(id="f"), args=[], keywords=[]), args=[], keywords=[])
  expr: CppNode = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "unknown"


def test_mapper_expression_call_attribute_complex() -> None:
  """Docstring."""
  mapper: ASTToCppMapper = ASTToCppMapper()
  # Call with an attribute where value is not a Name
  node: ast.Call = ast.Call(
    func=ast.Attribute(value=ast.Call(func=ast.Name(id="f"), args=[], keywords=[]), attr="method"), args=[], keywords=[]
  )
  expr: CppNode = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "method"


def test_mapper_expression_constant() -> None:
  """Docstring."""
  mapper: ASTToCppMapper = ASTToCppMapper()
  node: ast.Constant = ast.Constant(value=42)
  expr: CppNode = mapper.map_expression(node)
  assert isinstance(expr, Identifier)
  assert expr.name == "42"


def test_mapper_expression_unsupported() -> None:
  """Docstring."""
  mapper: ASTToCppMapper = ASTToCppMapper()
  node: ast.List = ast.List(elts=[])
  with pytest.raises(ValueError):
    mapper.map_expression(node)
