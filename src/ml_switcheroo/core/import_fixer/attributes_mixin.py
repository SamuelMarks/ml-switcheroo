"""Attribute Logic Mixin.

Handles the transformation of `Attribute` nodes, specifically for collapsing
fully qualified names into aliases (e.g. `jax.numpy.abs` -> `jnp.abs`) and
simplifying re-export chains.
"""

import libcst as cst

from ml_switcheroo.core.import_fixer.utils import REDUNDANT_SEGMENTS, get_root_name
from ml_switcheroo.core.scanners import get_full_name


class AttributeMixin(cst.CSTTransformer):
  """Mixin for processing Attribute nodes."""

  _path_to_alias: dict[str, str]
  _defined_names: set[str]
  target_framework: str | None

  def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
    """Handle path simplification and alias collapsing.

    Args:
        original_node: The node before transformation.
        updated_node: The node after transformation.

    Returns:
        The transformed expression node.

    """
    # 1. Deep Path Simplification (Re-export Cleanup)
    # Scenario: nnx.module.Module -> nnx.Module
    simplified = self._simplify_reexports(updated_node)
    if simplified is not updated_node:
      return simplified

    # 2. Alias Collapsing (Fully Qualified -> Alias)
    full_name = get_full_name(original_node)

    # Assuming self._path_to_alias is populated by BaseImportFixer
    try:
      path_to_alias = self._path_to_alias
    except AttributeError:
      path_to_alias = {}
    if not path_to_alias:
      return updated_node

    parts = full_name.split(".")
    # Try matching prefixes from longest to shortest
    for i in range(len(parts) - 1, 0, -1):
      prefix = ".".join(parts[:i])

      if prefix in path_to_alias:
        alias = str(path_to_alias[prefix])
        suffix_parts = parts[i:]

        # Construct collapsed node: alias.Remainder
        new_node: cst.Attribute = cst.Attribute(value=cst.Name(alias), attr=cst.Name(suffix_parts[0]))
        for part in suffix_parts[1:]:
          new_node = cst.Attribute(value=new_node, attr=cst.Name(part))

        # Apply deep simplification again on the collapsed result
        return self._simplify_reexports(new_node)

    return updated_node

  def _simplify_reexports(self, node: cst.Attribute) -> cst.BaseExpression:
    """Detect and strips redundant internal modules.

    E.g. ``nnx.module.Module`` becomes ``nnx.Module``.

    Args:
        node: The attribute node to inspect.

    Returns:
        The simplified attribute expression or the original node.

    """
    if not isinstance(node.value, cst.Attribute):
      return node

    middle_attr = node.value.attr.value
    if middle_attr not in REDUNDANT_SEGMENTS:
      return node

    # Safety Check: The root of this chain must be a known framework alias or import
    root_name = get_root_name(node)

    # Build list of safe roots (imported aliases or target frameworks)
    safe_roots: set[str] = set()
    try:
      defined_names = self._defined_names
      safe_roots.update(defined_names)
    except AttributeError:
      pass
    try:
      path_to_alias = self._path_to_alias
      safe_roots.update(str(v) for v in path_to_alias.values())
    except AttributeError:
      pass
    try:
      target_fw = self.target_framework
    except AttributeError:
      target_fw = None
    if isinstance(target_fw, str):
      safe_roots.add(target_fw)

    if root_name not in safe_roots:
      return node

    # Collapse: Remove the middle attribute
    new_base = node.value.value
    return node.with_changes(value=new_base)
