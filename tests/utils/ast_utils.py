"""AST Utilities for Test Verification.

This module provides utility functions and sentinel classes for comparing Python
Abstract Syntax Trees (ASTs) during test execution and verification. It enables
deep structural comparison of AST nodes, lists, and primitives, abstracting away
source positions, node identity, and formatting variations.

Classes:
    Undefined: A sentinel class representing an undefined or missing attribute.

Functions:
    cmp_ast: Recursively compares AST nodes, sequences of nodes, or values.
"""

from ast import AST


class Undefined:
  """Sentinel class representing undefined or missing attributes on AST nodes.

  This class acts as a placeholder value when querying fields on AST nodes
  that are absent or not defined, distinguishing them from fields explicitly
  set to None.
  """


def cmp_ast(node0, node1):
  """Recursively compares two AST nodes or collections of AST nodes for structural equality.

  This function performs a deep structural comparison of AST nodes, lists/tuples of
  nodes, and other primitive values. It ignores object identity, formatting variations,
  and source-code locations, focusing solely on node types, structures, and field values.

  Args:
      node0 (Any): The first AST node, sequence of nodes, or primitive value to compare.
      node1 (Any): The second AST node, sequence of nodes, or primitive value to compare.

  Returns:
      bool: True if node0 and node1 are structurally equivalent, False otherwise.

  """
  if type(node0) is not type(node1):
    return False
  if isinstance(node0, (list, tuple)):
    if len(node0) != len(node1):
      return False
    for left, right in zip(node0, node1):
      if not cmp_ast(left, right):
        return False
  elif isinstance(node0, AST):
    for field in node0._fields:
      left = getattr(node0, field, Undefined)
      right = getattr(node1, field, Undefined)
      if not cmp_ast(left, right):
        return False
  else:
    return node0 == node1
  return True
