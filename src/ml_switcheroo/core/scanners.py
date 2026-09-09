"""AST Scanners for Symbol Usage Detection.

This module provides LibCST visitors that analyze code to determine if specific
names, frameworks, or aliases are actively referenced in the source body.

These scanners are critical for the ``ImportFixer`` logic:

1.  If a framework alias (e.g., ``jnp``) is injected, ``SimpleNameScanner`` verifies
    it is actually used before committing the import.
2.  If a source import (e.g., ``import torch``) is slated for removal, ``UsageScanner``
    checks if it persists in the code (e.g., inside an Escape Hatch) to prevent
    breaking valid code.
"""

from typing import Set, Union

import libcst as cst


def get_full_name(node: Union[cst.Name, cst.Attribute, cst.BaseExpression]) -> str:
  """Recursively resolves a CST Name or Attribute chain to a dot-separated string.

  This helper flattens the AST representation of dotted names into strings
  comparable with import definitions.

  Args:
      node: The CST node representing the identifier. Typically a ``cst.Name`` (e.g., ``x``) or ``cst.Attribute`` (e.g., ``x.y``).

  Returns:
      The fully qualified string representation (e.g., "torch.nn.functional"). Returns an empty string if the node structure is not a supported Name/Attribute chain.
  """
  if isinstance(node, cst.Name):
    return node.value
  elif isinstance(node, cst.Attribute):
    prefix = get_full_name(node.value)
    return f"{prefix}.{node.attr.value}" if prefix else node.attr.value
  return ""


class SimpleNameScanner(cst.CSTVisitor):
  """Scan for the usage of a specific identifier in the code body.

  This visitor is designed to check for the presence of variables or aliases
  (like ``jnp``, ``tf``, ``mx``) *outside* of import statements. It is used to
  determine if a speculative import injection is actually required.
  """

  def __init__(self, target_name: str) -> None:
    """Initialize the scanner.

    Args:
        target_name: The string alias to search for.

    """
    self.target_name = target_name
    self.found = False
    self._in_import = False

  def visit_Import(self, node: cst.Import) -> None:
    """Flag entry into an ``import ...`` statement.

    Names appearing here are definitions, not usages.

    Args:
        node: The CST Import node being visited.

    """
    self._in_import = True

  def leave_Import(self, node: cst.Import) -> None:
    """Flag exit from an ``import ...`` statement.

    Args:
        node: The CST Import node being left.

    """
    self._in_import = False

  def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
    """Flag entry into a ``from ... import ...`` statement.

    Args:
        node: The CST ImportFrom node being visited.

    """
    self._in_import = True

  def leave_ImportFrom(self, node: cst.ImportFrom) -> None:
    """Flag exit from a ``from ... import ...`` statement.

    Args:
        node: The CST ImportFrom node being left.

    """
    self._in_import = False

  def visit_Name(self, node: cst.Name) -> None:
    """Check if the visited name matches the target.

    If the name matches ``target_name`` and we are NOT currently inside an import
    definition, we mark ``found = True``.

    Args:
        node: The name node being visited.

    """
    if not self._in_import and not self.found:
      if node.value == self.target_name:
        self.found = True

  def should_traverse(self, _node: cst.CSTNode) -> bool:
    """Optimization hook to stop traversal once found.

    Args:
        _node: The CST node whose children are about to be visited.

    Returns:
        bool: False if the target has already been found, effectively
        short-circuiting the rest of the AST traversal.

    """
    return not self.found


class GlobalUsageScanner(cst.CSTVisitor):
  """Scan the AST for all names used outside of import declarations.

  This is primarily used by the `ImportFixer` for Generalized Dead-Code
  Elimination (DCE) of unused imports.
  """

  def __init__(self) -> None:
    """Initialize the GlobalUsageScanner."""
    self.used_names: Set[str] = set()
    self._in_import = False

  def visit_Import(self, node: cst.Import) -> None:
    """Flag entry into an ``import ...`` statement.

    Args:
        node: The import node.
    """
    self._in_import = True

  def leave_Import(self, node: cst.Import) -> None:
    """Exit import scope.

    Args:
        node: The import node being left.
    """
    self._in_import = False

  def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
    """Flag entry into a ``from ... import ...`` statement.

    Args:
        node: The import-from node.
    """
    self._in_import = True

  def leave_ImportFrom(self, node: cst.ImportFrom) -> None:
    """Exit import-from scope.

    Args:
        node: The import-from node being left.
    """
    self._in_import = False

  def visit_Name(self, node: cst.Name) -> None:
    """Record any name used outside of an import block.

    Args:
        node: The name node.
    """
    if not self._in_import:
      self.used_names.add(node.value)
