"""Extra unit tests for the ASTToCppMapper.

This module contains additional unit tests for the ASTToCppMapper class.
It validates complex and boundary behavior when translating Python AST expression
nodes into C++ CST nodes. The tests specifically target less common or default
fallback paths for binary operations (such as subtraction, multiplication, division, and
unmapped operators defaulting to standard addition), constant representations, function calls
originating from unsupported non-name/non-attribute constructs (e.g., calling an integer),
and error-handling branches for unsupported AST node types.
"""

import ast
import typing
import pytest
from ml_switcheroo.core.compiler.backends.cpp.mapper import ASTToCppMapper
from ml_switcheroo.core.compiler.backends.cpp.cst import Expression, BinaryExpression, Identifier


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
