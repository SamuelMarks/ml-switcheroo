"""
Utilities for AST Node Construction and Inspection.

This module provides helper functions to convert Python runtime objects (dictionaries,
lists, primitives) into LibCST nodes, as well as utilities for inspecting import definitions.
"""

from typing import Any, Union
import json
import libcst as cst


def get_import_root(node: Union[cst.Name, cst.Attribute]) -> str:
  """
  Recursively extracts the root package name from a CST node.

  Args:
      node: The AST node representing the module name.

  Returns:
      str: The root package identifier (e.g. "scipy" from "scipy.special").
  """
  if isinstance(node, cst.Name):
    return node.value
  if isinstance(node, cst.Attribute):
    return get_import_root(node.value)
  return ""


def is_docstring(node: cst.CSTNode, idx: int) -> bool:
  """
  Checks if a statement node represents a module docstring.

  Args:
      node: The statement node to check.
      idx: The index of the statement within the module body.

  Returns:
      bool: True if it is a docstring (Expr containing String at index 0).
  """
  if idx != 0:
    return False
  if isinstance(node, cst.SimpleStatementLine) and len(node.body) == 1:
    expr = node.body[0]
    if isinstance(expr, cst.Expr) and isinstance(expr.value, (cst.SimpleString, cst.ConcatenatedString)):
      return True
  return False


def is_future_import(node: cst.CSTNode) -> bool:
  """
  Checks if a statement is a `from __future__ import ...` directive.

  Args:
      node: The statement node to check.

  Returns:
      bool: True if it is a future import.
  """
  if isinstance(node, cst.SimpleStatementLine):
    for stmt in node.body:
      if isinstance(stmt, cst.ImportFrom):
        if stmt.module and isinstance(stmt.module, cst.Name) and stmt.module.value == "__future__":
          return True
  return False


def convert_to_cst_literal(val: Any) -> cst.BaseExpression:
  """
  Recursively converts a python primitive or container to a CST node.
  Robustly handles strings using standard JSON encoding to prevent syntax errors
  and ensure double-quotes are used (matching test expectations).

  Args:
      val: The python value to convert.

  Returns:
      cst.BaseExpression: The literal node.
  """
  # 1. Container Recursion (List/Tuple)
  if isinstance(val, (list, tuple)):
    elements = []
    for item in val:
      node = convert_to_cst_literal(item)
      elements.append(cst.Element(value=node, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))

    if elements:
      # Strip trailing comma from last element for cleaner syntax
      last = elements[-1]
      elements[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    if isinstance(val, list):
      return cst.List(elements=elements)
    else:
      return cst.Tuple(elements=elements)

  # 2. Key-Value Recursion (Dict)
  if isinstance(val, dict):
    elements = []
    for k, v in val.items():
      k_node = convert_to_cst_literal(k)
      v_node = convert_to_cst_literal(v)
      elements.append(
        cst.DictElement(
          key=k_node,
          value=v_node,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    if elements:
      last = elements[-1]
      elements[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return cst.Dict(elements=elements)

  # 3. Primitives
  if isinstance(val, bool):
    return cst.Name("True") if val else cst.Name("False")
  elif isinstance(val, int):
    return cst.Integer(str(val))
  elif isinstance(val, float):
    # repr ensures high precision float string
    return cst.Float(repr(val))
  elif isinstance(val, str):
    # Use json.dumps to force double quotes for string literals,
    # matching expectation in tests and standard style.
    return cst.SimpleString(json.dumps(val))
  elif val is None:
    return cst.Name("None")
  else:
    # Final Fallback
    return cst.SimpleString(repr(str(val)))
