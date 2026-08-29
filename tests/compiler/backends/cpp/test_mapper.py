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
import typing

import pytest
from pytest import MonkeyPatch

from ml_switcheroo.core.compiler.backends.cpp.cst import BinaryExpression, Expression, Identifier
from ml_switcheroo.core.compiler.backends.cpp.mapper import ASTToCppMapper


def test_ast_to_cpp_mapper() -> None:
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
  tree: ast.expr = ast.parse("a + b", mode="eval").body
  cpp_expr: Expression = mapper.map_expression(tree)
  assert isinstance(cpp_expr, BinaryExpression)
  assert cpp_expr.left.name == "a"
  assert cpp_expr.operator == "+"
  assert cpp_expr.right.name == "b"
  assert cpp_expr.to_text() == "a + b"

  # Function call
  tree2: ast.expr = ast.parse("torch.matmul(x, y)", mode="eval").body
  cpp_expr2: Expression = mapper.map_expression(tree2)
  assert cpp_expr2.to_text() == "torch.matmul(x, y)"

  # Function call with Name
  tree3: ast.expr = ast.parse("len(x)", mode="eval").body
  cpp_expr3: Expression = mapper.map_expression(tree3)
  assert cpp_expr3.to_text() == "len(x)"

  # Function call with complex expression
  tree4: ast.expr = ast.parse("funcs[0](x)", mode="eval").body
  cpp_expr4: Expression = mapper.map_expression(tree4)
  assert cpp_expr4.to_text() == "unknown(x)"

  # Function call with complex Attribute
  tree5: ast.expr = ast.parse("a.b.c(x)", mode="eval").body
  cpp_expr5: Expression = mapper.map_expression(tree5)
  assert cpp_expr5.to_text() == "c(x)"


def test_ast_to_cpp_mapper_no_operators_json(monkeypatch: MonkeyPatch) -> None:
  """Docstring."""
  import os

  from ml_switcheroo.core.compiler.backends.cpp.mapper import ASTToCppMapper

  with monkeypatch.context() as m:
    m.setattr(os.path, "exists", lambda path: False)
    mapper = ASTToCppMapper()
    assert mapper.op_map == {}


# --- Merged from test_mapper_extra.py ---

"""Extra unit tests for the ASTToCppMapper.

This module contains additional unit tests for the ASTToCppMapper class.
It validates complex and boundary behavior when translating Python AST expression
nodes into C++ CST nodes. The tests specifically target less common or default
fallback paths for binary operations (such as subtraction, multiplication, division, and
unmapped operators defaulting to standard addition), constant representations, function calls
originating from unsupported non-name/non-attribute constructs (e.g., calling an integer),
and error-handling branches for unsupported AST node types.
"""


def test_ast_to_cpp_mapper_extra() -> None:
  """Verify mapping behavior for edge-case, fallback, and unsupported Python AST nodes.

  This test instantiates the ASTToCppMapper and executes several verification assertions:
  1. Validates that subtraction (`ast.Sub`), multiplication (`ast.Mult`), and division (`ast.Div`)
     binary operations map to their corresponding C++ operator representations (`-`, `*`, `/`).
     It also checks that unrecognized binary operators (such as bitwise OR, `ast.BitOr`) default
     gracefully to `+`.
  2. Verifies that constant values (e.g., `ast.Constant(value=42)`) are correctly wrapped as
     Identifier nodes in the generated C++ CST, using their string value representation.
  3. Evaluates mapping of call expressions (`ast.Call`) where the invoked function is neither a
     name nor an attribute (e.g., a constant integer literal wrapper), verifying that the
     mapper defaults the C++ function name representation to "unknown".
  4. Confirms that attempting to map an unsupported node type (such as `ast.Pass`) correctly
     triggers and raises a ValueError.

  Args:
      None.

  Returns:
      None.
  """
  mapper = ASTToCppMapper()

  # Sub, Mult, Div
  for op_str, op_node in [("-", ast.Sub()), ("*", ast.Mult()), ("/", ast.Div()), ("+", ast.BitOr())]:
    expr: ast.expr = ast.BinOp(left=ast.Name(id="a"), op=op_node, right=ast.Name(id="b"))
    cpp_expr: Expression = mapper.map_expression(expr)
    expected_op: str = "+" if isinstance(op_node, ast.BitOr) else op_str
    assert isinstance(cpp_expr, BinaryExpression)
    assert cpp_expr.operator == expected_op

  # Constant
  expr = ast.Constant(value=42)
  cpp_expr = mapper.map_expression(expr)
  assert isinstance(cpp_expr, Identifier)
  assert cpp_expr.name == "42"

  # Call with non-name/attribute func (e.g. lambda)
  # Just construct a weird AST node
  expr = ast.Call(func=ast.Constant(value=1), args=[], keywords=[])
  cpp_expr = mapper.map_expression(expr)
  # The generated expression should be a FunctionCall, its name property would be 'unknown'
  assert getattr(cpp_expr, "name", None) == "unknown"

  # Unsupported
  with pytest.raises(ValueError):
    mapper.map_expression(typing.cast(ast.expr, ast.Pass()))
