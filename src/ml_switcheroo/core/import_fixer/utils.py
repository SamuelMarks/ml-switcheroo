"""
Utilities for the Import Fixer.

Contains static helper functions for analyzing AST nodes, extracting names,
generating signatures for deduplication, and creating CST nodes.
"""

from typing import Union

import libcst as cst
from ml_switcheroo.utils.node_diff import capture_node_source

# Segments that should be collapsed if they appear in framework paths.
# e.g., 'flax.nnx.module.Module' -> 'module' is redundant if 'Module' is exported by 'nnx'.
REDUNDANT_SEGMENTS = {"module", "_src", "src"}


def get_root_name(node: Union[cst.Name, cst.Attribute]) -> str:
  """
  Recursively extracts the root identifier from a Name or Attribute chain.

  Args:
      node: The CST node (e.g., `Attribute(value=Name('torch'), ...)`).

  Returns:
      str: The root name (e.g., "torch").
  """
  if isinstance(node, cst.Name):
    return node.value
  if isinstance(node, cst.Attribute):
    return get_root_name(node.value)
  return ""


def create_dotted_name(name_str: str) -> Union[cst.Name, cst.Attribute]:
  """
  Creates a CST node structure for a dotted path string.

  Args:
      name_str (str): Dot-separated path (e.g. "jax.numpy").

  Returns:
      Union[cst.Name, cst.Attribute]: The constructed AST node.
  """
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def get_signature(node: cst.CSTNode) -> str:
  """
  Computes a deduplication signature for an AST node (usually an Import statement).

  It normalizes the source code representation to ignore basic formatting differences,
  allowing detection of duplicate import injections.

  Args:
      node: The CST node to sign.

  Returns:
      str: Normalized source code string.
  """
  target = node
  # Unwrap simple statement lines if necessary to get content
  while isinstance(target, cst.SimpleStatementLine) and len(target.body) > 0:
    target = target.body[0]

  src = capture_node_source(target)
  return " ".join(src.split())


def is_docstring(node: cst.CSTNode, idx: int) -> bool:
  """
  Determines if a statement node represents a module docstring.

  Args:
      node: The statement node from the module body.
      idx: The index of this statement in the body list.

  Returns:
      bool: True if it is a docstring (string expression at index 0).
  """
  if idx != 0:
    return False
  if isinstance(node, cst.SimpleStatementLine):
    if len(node.body) == 1 and isinstance(node.body[0], cst.Expr):
      expr = node.body[0].value
      if isinstance(expr, (cst.SimpleString, cst.ConcatenatedString)):
        return True
  return False


def is_future_import(node: cst.CSTNode) -> bool:
  """
  Determines if a statement is a `from __future__ import ...` directive.

  Args:
      node: The statement node.

  Returns:
      bool: True if it is a future import.
  """
  if isinstance(node, cst.SimpleStatementLine):
    for small_stmt in node.body:
      if isinstance(small_stmt, cst.ImportFrom):
        if small_stmt.module and isinstance(small_stmt.module, cst.Name):
          if small_stmt.module.value == "__future__":
            return True
  return False
