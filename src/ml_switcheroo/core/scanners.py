"""
AST Scanners for Symbol Usage Detection.

This module provides LibCST visitors that analyze code to determine if specific
names, frameworks, or aliases are actively referenced in the source body.

These scanners are critical for the `ImportFixer` logic:
1.  If a framework alias (e.g., `jnp`) is injected, `SimpleNameScanner` verifies
    it is actually used before committing the import.
2.  If a source import (e.g., `import torch`) is slated for removal, `UsageScanner`
    checks if it persists in the code (e.g., inside an Escape Hatch) to prevent
    breaking valid code.
"""

from typing import Set, Union

import libcst as cst


def get_full_name(node: Union[cst.Name, cst.Attribute]) -> str:
  """
  Recursively resolves a CST Name or Attribute chain to a dot-separated string.

  This helper flattens the AST representation of dotted names into strings
  comparable with import definitions.

  Args:
    node: The CST node representing the identifier.
      Typically a `cst.Name` (e.g., `x`) or `cst.Attribute` (e.g., `x.y`).

  Returns:
    str: The fully qualified string representation (e.g., "torch.nn.functional").
    Returns an empty string if the node structure is not a supported Name/Attribute chain.

  Example:
    >>> get_full_name(cst.Attribute(value=cst.Name("torch"), attr=cst.Name("abs")))
    'torch.abs'
  """
  if isinstance(node, cst.Name):
    return node.value
  elif isinstance(node, cst.Attribute):
    return f"{get_full_name(node.value)}.{node.attr.value}"
  return ""


class SimpleNameScanner(cst.CSTVisitor):
  """
  Scans for the usage of a specific identifier in the code body.

  This visitor is designed to check for the presence of variables or aliases
  (like `jnp`, `tf`, `mx`) *outside* of import statements. It is used to
  determine if a speculative import injection is actually required.

  Attributes:
    target_name (str): The identifier string to search for (e.g., "jnp").
    found (bool): State flag, set to True immediately upon finding a match.
    _in_import (bool): Internal state tracking if traversal is currently
      inside an Import/ImportFrom node.
  """

  def __init__(self, target_name: str) -> None:
    """
    Initializes the scanner.

    Args:
      target_name: The string alias to search for.
    """
    self.target_name = target_name
    self.found = False
    self._in_import = False

  def visit_Import(self, node: cst.Import) -> None:
    """
    Flags entry into an `import ...` statement.
    Names appearing here are definitions, not usages.
    """
    self._in_import = True

  def leave_Import(self, node: cst.Import) -> None:
    """Flags exit from an `import ...` statement."""
    self._in_import = False

  def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
    """
    Flags entry into a `from ... import ...` statement.
    """
    self._in_import = True

  def leave_ImportFrom(self, node: cst.ImportFrom) -> None:
    """Flags exit from a `from ... import ...` statement."""
    self._in_import = False

  def visit_Name(self, node: cst.Name) -> None:
    """
    Checks if the visited name matches the target.

    If the name matches `target_name` and we are NOT currently inside an import
    definition, we mark `found = True`.

    Args:
      node: The name node being visited.
    """
    if not self._in_import and not self.found:
      if node.value == self.target_name:
        self.found = True

  def should_traverse(self, _node: cst.CSTNode) -> bool:
    """
    Optimization hook to stop traversal once found.

    Returns:
      bool: False if the target has already been found, effectively
      short-circuiting the rest of the AST traversal.
    """
    return not self.found


class UsageScanner(cst.CSTVisitor):
  """
  Scans the AST for usages of a specific framework root or its local aliases.

  This class implements a multi-pass logic during a single traversal:
  1.  **Cataloging**: It identifies all aliases bound to the `source_fw`
      (e.g., `import torch as t` -> `t` is an alias).
  2.  **Detection**: It checks if any of these cataloged aliases are used
      in the code body (e.g., `t.abs(x)`).

  This is primarily used by the `ImportFixer` to decide whether to prune the
  original import statement. If usages persist (e.g., via the Escape Hatch),
  the import MUST be preserved to keep the code valid.

  Attributes:
    source_fw (str): The root framework name to scan for (e.g., 'torch').
    found_usages (Set[str]): A set of specific aliases that were visibly used.
    _tracked_aliases (Set[str]): Internal registry of names that resolve to
      the source framework. Initialized with `[source_fw]`.
    _in_import (bool): Internal state tracking import block context.
  """

  def __init__(self, source_fw: str) -> None:
    """
    Initializes the UsageScanner.

    Args:
      source_fw: The framework string (e.g., 'torch').
    """
    self.source_fw = source_fw
    self.found_usages: Set[str] = set()

    # We track the root name itself, plus any aliases found during visit_Import
    self._tracked_aliases: Set[str] = {source_fw}
    self._in_import = False

  def get_result(self) -> bool:
    """
    Returns the scan result.

    Returns:
      bool: True if any tracked alias was found used in the body.
    """
    return len(self.found_usages) > 0

  def visit_Import(self, node: cst.Import) -> None:
    """
    Catalogs names bound by `import ...`.

    Logic:
      - `import torch` -> tracks 'torch'.
      - `import torch as t` -> tracks 't'.
      - `import torch.nn as nn` -> tracks 'nn' (because it stems from torch).

    Args:
      node: The import node.
    """
    self._in_import = True
    for alias in node.names:
      base_name = get_full_name(alias.name)

      # Check if this import is relevant (matches source_fw or source_fw.*)
      if base_name == self.source_fw or base_name.startswith(f"{self.source_fw}."):
        # Determine the bound name in local scope
        if alias.asname:
          bound_name = alias.asname.name.value
        else:
          # For `import torch.nn`, the bound name is the top-level 'torch'
          # For `import torch`, the bound name is 'torch'
          bound_name = base_name.split(".")[0]

        self._tracked_aliases.add(bound_name)

  def leave_Import(self, node: cst.Import) -> None:
    """Exit import scope."""
    self._in_import = False

  def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
    """
    Catalogs names bound by `from ... import ...`.

    Logic:
      - `from torch import nn` -> tracks 'nn'.
      - `from torch.nn import Linear` -> tracks 'Linear'.

    Args:
      node: The import-from node.
    """
    self._in_import = True
    if not node.module:
      return

    module_name = get_full_name(node.module)

    # Check relevancy
    if module_name == self.source_fw or module_name.startswith(f"{self.source_fw}."):
      for alias in node.names:
        if isinstance(alias, cst.ImportAlias):
          if alias.asname:
            bound_name = alias.asname.name.value
          else:
            bound_name = alias.name.value
          self._tracked_aliases.add(bound_name)

  def leave_ImportFrom(self, node: cst.ImportFrom) -> None:
    """Exit import-from scope."""
    self._in_import = False

  def visit_Name(self, node: cst.Name) -> None:
    """
    Checks if a name in the body matches one of our tracked aliases.

    If found, it is recorded in `found_usages`.

    Args:
      node: The name node.
    """
    if not self._in_import:
      if node.value in self._tracked_aliases:
        self.found_usages.add(node.value)
