"""Unit tests for `ASTToCppMapper` (`ml_switcheroo.core.compiler.backends.cpp.mapper`)."""

import ast
import os

import pytest

from ml_switcheroo.core.compiler.backends.cpp.cst import BinaryExpression, Identifier, MethodCall
from ml_switcheroo.core.compiler.backends.cpp.mapper import ASTToCppMapper


def test_ast_to_cpp_mapper_operators_calls_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ASTToCppMapper` maps binary ops, calls, constants, missing `operators.json`, and unsupported nodes.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  mapper = ASTToCppMapper()

  # Binary operations (known `+`, `-`, `*`, `/` and fallback `+` on unknown `BitOr`)
  for op_str, op_node in (("+", ast.Add()), ("-", ast.Sub()), ("*", ast.Mult()), ("/", ast.Div()), ("+", ast.BitOr())):
    cpp_bin = mapper.map_expression(ast.BinOp(left=ast.Name(id="a"), op=op_node, right=ast.Name(id="b")))
    assert isinstance(cpp_bin, BinaryExpression)
    assert cpp_bin.operator == op_str

  assert mapper.map_expression(ast.parse("a + b", mode="eval").body).to_text() == "a + b"
  assert mapper.map_expression(ast.parse("torch.matmul(x, y)", mode="eval").body).to_text() == "torch.matmul(x, y)"
  assert mapper.map_expression(ast.parse("len(x)", mode="eval").body).to_text() == "len(x)"
  assert mapper.map_expression(ast.parse("funcs[0](x)", mode="eval").body).to_text() == "unknown(x)"
  assert mapper.map_expression(ast.parse("a.b.c(x)", mode="eval").body).to_text() == "c(x)"

  # Constant and call with non-Name/non-Attribute target
  const_expr = mapper.map_expression(ast.Constant(value=42))
  assert isinstance(const_expr, Identifier)
  assert const_expr.name == "42"

  unknown_call = mapper.map_expression(ast.Call(func=ast.Constant(value=1), args=[], keywords=[]))
  assert isinstance(unknown_call, MethodCall)
  assert unknown_call.name == "unknown"

  with pytest.raises(ValueError, match="Unsupported AST node"):
    mapper.map_expression(ast.List(elts=[]))

  # Missing `operators.json` initializes `op_map = {}` cleanly
  with monkeypatch.context() as mp:
    mp.setattr(os.path, "exists", lambda _p: False)
    empty_mapper = ASTToCppMapper()
    assert empty_mapper.op_map == {}
