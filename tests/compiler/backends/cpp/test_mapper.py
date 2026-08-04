"""Unit tests for the C++ compiler backend AST mapper.

This module contains test cases to verify the translation of Python AST structures
into equivalent C++ Concrete Syntax Tree (CST) nodes using the `ASTToCppMapper`
component. It ensures correct handling of binary expressions, various function
calls (including simple calls, attributes, nested subscripts, etc.), and their
subsequent text serialization.

Attributes:
    None
"""

import ast
from ml_switcheroo.core.compiler.backends.cpp.mapper import ASTToCppMapper
from ml_switcheroo.core.compiler.backends.cpp.cst import BinaryExpression


def test_ast_to_cpp_mapper():
  """Tests mapping of various Python AST expressions to C++ representations.

  This test case validates the conversion of distinct AST nodes such as binary
  operators (e.g., addition) and different forms of function calls (e.g., basic name calls,
  attribute calls, complex subscripts/attribute chains) into correct C++ CST structures.
  It verifies both the structural types of the mapped outputs and their serialized
  C++ text representation.

  Args:
      None

  Returns:
      None: This test function returns nothing but asserts correct behavior.
  """
  mapper = ASTToCppMapper()

  # Simple addition
  tree = ast.parse("a + b", mode="eval").body
  cpp_expr = mapper.map_expression(tree)
  assert isinstance(cpp_expr, BinaryExpression)
  assert cpp_expr.left.name == "a"
  assert cpp_expr.operator == "+"
  assert cpp_expr.right.name == "b"
  assert cpp_expr.to_text() == "a + b"

  # Function call
  tree2 = ast.parse("torch.matmul(x, y)", mode="eval").body
  cpp_expr2 = mapper.map_expression(tree2)
  assert cpp_expr2.to_text() == "torch.matmul(x, y)"

  # Function call with Name
  tree3 = ast.parse("len(x)", mode="eval").body
  cpp_expr3 = mapper.map_expression(tree3)
  assert cpp_expr3.to_text() == "len(x)"

  # Function call with complex expression
  tree4 = ast.parse("funcs[0](x)", mode="eval").body
  cpp_expr4 = mapper.map_expression(tree4)
  assert cpp_expr4.to_text() == "unknown(x)"

  # Function call with complex Attribute
  tree5 = ast.parse("a.b.c(x)", mode="eval").body
  cpp_expr5 = mapper.map_expression(tree5)
  assert cpp_expr5.to_text() == "c(x)"
