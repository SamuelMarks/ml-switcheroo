"""
Symbol Resolution Mixin.

Handles the tracking of Python imports and aliases to resolve AST nodes (names and attributes)
to their fully qualified API paths (e.g. `t.abs` -> `torch.abs`).
"""

from typing import Optional, Dict
import libcst as cst


class ResolverMixin:
  """
  Mixin for resolving AST nodes to fully qualified names based on imports.

  Assumed attributes on self:
      _alias_map (Dict[str, str]): Mapping of local aliases to full paths.
  """

  def _get_qualified_name(self, node: cst.BaseExpression) -> Optional[str]:
    """
    Resolves a CST node to its fully qualified name using import aliases.

    Example:
        If ``import torch.nn as nn`` exists, ``nn.Linear`` resolves to ``torch.nn.Linear``.

    Args:
        node: The CST expression (Name or Attribute).

    Returns:
        Optional[str]: The resolved string (e.g. 'torch.abs') or None if unresolvable.
    """
    full_str = self._cst_to_string(node)
    if not full_str:
      return None

    parts = full_str.split(".")
    root = parts[0]

    if root in self._alias_map:
      canonical_root = self._alias_map[root]
      if len(parts) > 1:
        # e.g. root='nn' -> 'torch.nn', parts=['nn', 'Linear'] -> 'torch.nn.Linear'
        return f"{canonical_root}.{'.'.join(parts[1:])}"
      return canonical_root

    return full_str

  def _cst_to_string(self, node: cst.BaseExpression) -> Optional[str]:
    """
    Helper to flatten Attribute chains into strings.

    Args:
        node: The CST node to stringify.

    Returns:
        Optional[str]: Dotted string path (e.g. "a.b.c") or None if complex.
    """
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.BinaryOperation):
      return type(node.operator).__name__
    elif isinstance(node, cst.Attribute):
      base = self._cst_to_string(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    return None

  def visit_Import(self, node: cst.Import) -> Optional[bool]:
    """
    Scans ``import ...`` statements to populate the alias map.
    Example: ``import torch.nn as nn`` -> ``_alias_map['nn'] = 'torch.nn'``.
    """
    for alias in node.names:
      full_name = self._cst_to_string(alias.name)
      if not full_name:
        continue

      if alias.asname:
        local_name = alias.asname.name.value
        self._alias_map[local_name] = full_name
      else:
        root = full_name.split(".")[0]
        self._alias_map[root] = root
    return False

  def visit_ImportFrom(self, node: cst.ImportFrom) -> Optional[bool]:
    """
    Scans ``from ... import ...`` statements to populate the alias map.
    Example: ``from torch import nn`` -> ``_alias_map['nn'] = 'torch.nn'``.
    """
    if node.relative:
      return False

    module_name = self._cst_to_string(node.module) if node.module else ""
    if not module_name:
      return False

    if isinstance(node.names, cst.ImportStar):
      return False

    for alias in node.names:
      if not isinstance(alias, cst.ImportAlias):
        continue

      imported_name = alias.name.value
      canonical_source = f"{module_name}.{imported_name}"

      if alias.asname:
        local_name = alias.asname.name.value
      else:
        local_name = imported_name

      self._alias_map[local_name] = canonical_source

    return False
