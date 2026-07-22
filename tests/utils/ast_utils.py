"""Test suite for the Ast Utils module."""

from ast import AST


class Undefined:
  """Test suite for the Undefined component."""


def cmp_ast(node0, node1):
  """Helper to cmp AST."""
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
