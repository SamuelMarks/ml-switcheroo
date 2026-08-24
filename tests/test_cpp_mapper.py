"""Test module."""

import pytest
import ast
import os
from unittest.mock import patch, mock_open

from ml_switcheroo.core.compiler.backends.cpp.mapper import ASTToCppMapper
from ml_switcheroo.core.compiler.backends.cpp.cst import Identifier, BinaryExpression, MethodCall


def test_mapper_init_no_file(monkeypatch):
  """Test element."""
  monkeypatch.setattr(os.path, "exists", lambda p: False)
  mapper = ASTToCppMapper()
  assert mapper.op_map == {}


def test_mapper_init_with_file():
  """Test element."""
  mock_data = '{"Add": "+"}'
  with patch("os.path.exists", return_value=True), patch("builtins.open", mock_open(read_data=mock_data)):
    mapper = ASTToCppMapper()
    assert mapper.op_map == {"Add": "+"}


def test_mapper_expression_name():
  """Test element."""
  mapper = ASTToCppMapper()
  node = ast.Name(id="foo")
  expr = mapper.map_expression(node)
  assert isinstance(expr, Identifier)
  assert expr.name == "foo"


def test_mapper_expression_binop():
  """Test element."""
  mapper = ASTToCppMapper()
  mapper.op_map = {"Add": "+"}
  node = ast.BinOp(left=ast.Name(id="a"), op=ast.Add(), right=ast.Name(id="b"))
  expr = mapper.map_expression(node)
  assert isinstance(expr, BinaryExpression)
  assert expr.operator == "+"
  assert expr.left.name == "a"
  assert expr.right.name == "b"


def test_mapper_expression_binop_unknown_op():
  """Test element."""
  mapper = ASTToCppMapper()
  mapper.op_map = {}
  node = ast.BinOp(left=ast.Name(id="a"), op=ast.Sub(), right=ast.Name(id="b"))
  expr = mapper.map_expression(node)
  assert isinstance(expr, BinaryExpression)
  assert expr.operator == "+"  # Default


def test_mapper_expression_call_name():
  """Test element."""
  mapper = ASTToCppMapper()
  node = ast.Call(func=ast.Name(id="foo"), args=[ast.Name(id="a")], keywords=[])
  expr = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "foo"
  assert len(expr.arguments) == 1
  assert expr.arguments[0].name == "a"


def test_mapper_expression_call_attribute():
  """Test element."""
  mapper = ASTToCppMapper()
  node = ast.Call(func=ast.Attribute(value=ast.Name(id="obj"), attr="method"), args=[], keywords=[])
  expr = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "obj.method"


def test_mapper_expression_call_unknown():
  """Test element."""
  mapper = ASTToCppMapper()
  # Call with a function that is not Name or Attribute (e.g. Call of Call)
  node = ast.Call(func=ast.Call(func=ast.Name(id="f"), args=[], keywords=[]), args=[], keywords=[])
  expr = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "unknown"


def test_mapper_expression_call_attribute_complex():
  """Test element."""
  mapper = ASTToCppMapper()
  # Call with an attribute where value is not a Name
  node = ast.Call(
    func=ast.Attribute(value=ast.Call(func=ast.Name(id="f"), args=[], keywords=[]), attr="method"), args=[], keywords=[]
  )
  expr = mapper.map_expression(node)
  assert isinstance(expr, MethodCall)
  assert expr.name == "method"


def test_mapper_expression_constant():
  """Test element."""
  mapper = ASTToCppMapper()
  node = ast.Constant(value=42)
  expr = mapper.map_expression(node)
  assert isinstance(expr, Identifier)
  assert expr.name == "42"


def test_mapper_expression_unsupported():
  """Test element."""
  mapper = ASTToCppMapper()
  node = ast.List(elts=[])
  with pytest.raises(ValueError):
    mapper.map_expression(node)
