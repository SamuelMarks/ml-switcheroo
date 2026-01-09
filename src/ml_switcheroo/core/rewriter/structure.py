"""
Structure Rewriter Aggregator and Stage Definition.

This module defines the `StructureStage`, a standalone `RewriterStage` responsible
for structural transformations of Python code (classes, functions, and type hints).
It aggregates logic from specialized mixins.
"""

from typing import Optional, Dict, Any
import libcst as cst

from ml_switcheroo.core.rewriter.base import RewriterStage
from ml_switcheroo.core.rewriter.structure_class import ClassStructureMixin
from ml_switcheroo.core.rewriter.structure_func import FuncStructureMixin
from ml_switcheroo.core.rewriter.structure_types import TypeStructureMixin
from ml_switcheroo.core.escape_hatch import EscapeHatch


class StructureStage(
  ClassStructureMixin,
  FuncStructureMixin,
  TypeStructureMixin,
  RewriterStage,
):
  """
  Standalone transformer stage for structural rewrites.

  Handles:
  1. Class Inheritance swapping (e.g. torch.nn.Module -> flax.nnx.Module).
  2. Function Signature injection (e.g. adding 'rngs').
  3. Type Hint remapping (e.g. torch.Tensor -> jax.Array).

  Acts as the context-aware bridge for the sub-mixins.
  Inherits `RewriterProxy` attributes getters/setters via `RewriterStage`.
  """

  # --- Shared Helpers required by Mixins (Proxy Implementation) ---

  def _enter_scope(self) -> None:
    """Push a new scope onto the context stack."""
    self._scope_stack.append(set())

  def _exit_scope(self) -> None:
    """Pop the current scope from the context stack."""
    if len(self._scope_stack) > 1:
      self._scope_stack.pop()

  def _report_failure(self, reason: str) -> None:
    """Register a failure in the context."""
    self._current_stmt_errors.append(reason)

  def _cst_to_string(self, node: cst.BaseExpression) -> Optional[str]:
    """
    Helper to flatten Attribute chains into strings.
    """
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.Attribute):
      base = self._cst_to_string(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    return None

  def _get_qualified_name(self, node: cst.BaseExpression) -> Optional[str]:
    """
    Resolves a CST node to its fully qualified name using context aliases.
    """
    full_str = self._cst_to_string(node)
    if not full_str:
      return None

    parts = full_str.split(".")
    root = parts[0]

    if root in self._alias_map:
      canonical_root = self._alias_map[root]
      if len(parts) > 1:
        return f"{canonical_root}.{'.'.join(parts[1:])}"
      return canonical_root

    return full_str

  def _get_mapping(self, name: str, silent: bool = False) -> Optional[Dict[str, Any]]:
    """
    Queries Semantic Knowledge Base for a mapping.
    """
    lookup = self.semantics.get_definition(name)
    if not lookup:
      return None

    abstract_id, details = lookup
    target_impl = self.semantics.resolve_variant(abstract_id, self.target_fw)
    return target_impl

  def _create_name_node(self, api_path: str) -> cst.BaseExpression:
    """
    Creates a LibCST node structure from a dotted string.
    """
    parts = api_path.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))
    return node

  def _create_dotted_name(self, name_str: str) -> cst.BaseExpression:
    """Alias for _create_name_node."""
    return self._create_name_node(name_str)


# Legacy Mixin export for PivotRewriter compatibility
StructureMixin = StructureStage
