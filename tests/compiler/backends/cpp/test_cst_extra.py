"""Tests for C++ CST extra functionality."""

import pytest
from ml_switcheroo.core.compiler.backends.cpp.cst import CppNode, IncludeDirective, CppModule, Expression
from ml_switcheroo.core.compiler.backends.cpp.mapper import ASTToCppMapper
import ast


def test_cst_not_implemented() -> None:
  """Test that CppNode.to_text raises NotImplementedError for abstract nodes."""

  class Dummy(CppNode):
    """Dummy class."""

    pass

  with pytest.raises(NotImplementedError):
    Dummy().to_text()


def test_include_directive_validation() -> None:
  """Test that IncludeDirective validates its arguments."""
  with pytest.raises(ValueError):
    IncludeDirective(path=123)  # type: ignore
  with pytest.raises(ValueError):
    IncludeDirective(path="<foo>")


def test_cst_parse_method() -> None:
  """Test that CppNode.parse correctly parses code."""
  code: str = "int x = 5;"
  node: CppNode = CppNode.parse(code)
  assert isinstance(node, CppModule)


def test_ast_mapper_call_attr() -> None:
  """Test that ASTToCppMapper correctly maps chained attributes in calls."""
  # Covers line 37 in mapper.py (naive attribute translation)
  mapper = ASTToCppMapper()
  tree: ast.expr = ast.parse("a.b.c()", mode="eval").body
  cpp_expr: Expression = mapper.map_expression(tree)
  assert cpp_expr.to_text() == "c()"
