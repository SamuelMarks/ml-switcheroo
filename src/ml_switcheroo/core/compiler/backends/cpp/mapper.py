"""AST to C++ CST Mapper.

Maps standard Python AST nodes (used as semantic operations) to C++ CST nodes.
"""

import ast
import json
import os

from ml_switcheroo.core.compiler.backends.cpp.cst import (
  Expression,
  Identifier,
  BinaryExpression,
  MethodCall,
)


class ASTToCppMapper:
  """Translates Python AST nodes to CppNode representations.

  This mapper parses standard Python Abstract Syntax Tree (AST) expressions
  and converts them into their corresponding C++ Concrete Syntax Tree (CST)
  representations.
  """

  def __init__(self) -> None:
    """Initializes the AST mapper and loads the operator mappings."""
    self.op_map = {}
    operators_json_path = os.path.join(os.path.dirname(__file__), "operators.json")
    if os.path.exists(operators_json_path):
      with open(operators_json_path, "r", encoding="utf-8") as f:
        self.op_map = json.load(f)

  def map_expression(self, node: ast.expr) -> Expression:
    """Maps a Python AST expression to a C++ CST Expression.

    Args:
        node: The Python AST expression node to map.

    Returns:
        Expression: The mapped C++ CST Expression node.

    Raises:
        ValueError: If the AST node type is not supported.
    """
    if isinstance(node, ast.Name):
      return Identifier(name=node.id)
    elif isinstance(node, ast.BinOp):
      left = self.map_expression(node.left)
      right = self.map_expression(node.right)
      op_name = type(node.op).__name__
      op = self.op_map.get(op_name, "+")
      return BinaryExpression(left=left, operator=op, right=right)
    elif isinstance(node, ast.Call):
      if isinstance(node.func, ast.Name):
        func_name = node.func.id
      elif isinstance(node.func, ast.Attribute):
        # naive attribute translation
        func_name = f"{node.func.value.id}.{node.func.attr}" if isinstance(node.func.value, ast.Name) else node.func.attr
      else:
        func_name = "unknown"
      args = [self.map_expression(arg) for arg in node.args]
      return MethodCall(name=func_name, arguments=args)
    elif isinstance(node, ast.Constant):
      return Identifier(name=str(node.value))

    raise ValueError(f"Unsupported AST node: {type(node)}")
