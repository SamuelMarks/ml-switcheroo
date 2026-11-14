"""Attribute Alias Simplifier Collaborator.

Handles the transformation of ``cst.Attribute`` nodes for collapsing fully
qualified names into imported aliases (e.g. ``jax.numpy.abs`` -> ``jnp.abs``)
and simplifying redundant internal re-export chains (e.g. ``nnx.module.Module`` -> ``nnx.Module``).
"""

import libcst as cst

from ml_switcheroo.core.import_fixer.base import ImportFixerState
from ml_switcheroo.core.import_fixer.utils import REDUNDANT_SEGMENTS, get_root_name
from ml_switcheroo.core.scanners import get_full_name


class AttributeAliasSimplifier:
  """Collaborator simplifying ``cst.Attribute`` re-export chains and collapsing aliases.

  Attributes:
      state: Shared ``ImportFixerState`` instance.
  """

  def __init__(self, state: ImportFixerState) -> None:
    """Initialize the attribute simplifier with shared ``ImportFixerState``.

    Args:
        state: Shared ``ImportFixerState`` instance.
    """
    self.state: ImportFixerState = state

  def rewrite_attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
    """Handle path simplification and alias collapsing on ``cst.Attribute`` nodes.

    Args:
        original_node: CST node prior to child traversal.
        updated_node: CST node after child traversal.

    Returns:
        Simplified or alias-collapsed ``cst.BaseExpression``.
    """
    simplified = self.simplify_reexports(updated_node)
    if simplified is not updated_node:
      return simplified

    path_to_alias = self.state.path_to_alias
    if not path_to_alias:
      return updated_node

    full_name = get_full_name(original_node)
    parts = full_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
      prefix = ".".join(parts[:i])
      if prefix in path_to_alias:
        alias = str(path_to_alias[prefix])
        suffix_parts = parts[i:]

        new_node: cst.Attribute = cst.Attribute(value=cst.Name(alias), attr=cst.Name(suffix_parts[0]))
        for part in suffix_parts[1:]:
          new_node = cst.Attribute(value=new_node, attr=cst.Name(part))

        return self.simplify_reexports(new_node)

    return updated_node

  def simplify_reexports(self, node: cst.Attribute) -> cst.BaseExpression:
    """Strip redundant internal module segments from safe framework roots.

    Args:
        node: Attribute node to inspect.

    Returns:
        Simplified attribute expression or unchanged ``node``.
    """
    if not isinstance(node.value, cst.Attribute):
      return node

    middle_attr = node.value.attr.value
    if middle_attr not in REDUNDANT_SEGMENTS:
      return node

    root_name = get_root_name(node)

    safe_roots: set[str] = set(self.state.defined_names)
    safe_roots.update(str(v) for v in self.state.path_to_alias.values())
    if isinstance(self.state.target_framework, str):
      safe_roots.add(self.state.target_framework)

    if root_name not in safe_roots:
      return node

    return node.with_changes(value=node.value.value)


__all__ = ["AttributeAliasSimplifier"]
