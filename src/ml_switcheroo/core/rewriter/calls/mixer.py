"""
API Stage Definition (Call & Attribute Mixer).

Combines various mixins required to handle assignment operations, function
invocations, and attribute access in the AST.

This module exports `ApiStage`, which aggregates:
- :class:`InvocationMixin`: Helper logic for complex Call handling.
- :class:`AssignmentUnwrapMixin`: Logic for unwrapping functional return tuples.
- :class:`AttributeMixin`: Logic for rewriting attributes (constants/properties).
- :class:`ResolverMixin`: Logic for resolving aliases.
- :class:`ScopingMixin`: Logic for tracking stateful variables.

Note: `CallMixin` is provided as an alias for backward compatibility.
"""

from typing import Optional, Dict, Any, Union

import libcst as cst

from ml_switcheroo.core.rewriter.base import RewriterStage
from ml_switcheroo.core.rewriter.calls.assignment import AssignmentUnwrapMixin
from ml_switcheroo.core.rewriter.calls.invocation import InvocationMixin
from ml_switcheroo.core.rewriter.attributes import AttributeMixin

# Support Mixins
from ml_switcheroo.core.rewriter.resolver import ResolverMixin
from ml_switcheroo.core.rewriter.scopes import ScopingMixin
from ml_switcheroo.core.rewriter.ver_check import VersioningMixin
from ml_switcheroo.core.rewriter.errors import ErrorHandlingMixin
from ml_switcheroo.core.tracer import get_tracer


class ApiStage(
  InvocationMixin,
  AssignmentUnwrapMixin,
  AttributeMixin,
  ResolverMixin,
  ScopingMixin,
  ErrorHandlingMixin,
  VersioningMixin,
  RewriterStage,
):
  """
  Consolidated Rewriter Stage for API transformations.

  Handles:
  1.  **Function Calls**: `torch.abs(x)` -> `jax.numpy.abs(x)`.
  2.  **Attributes**: `torch.float32` -> `jax.numpy.float32`.
  3.  **Assignments**: Unwrapping functional returns for OOP consistency.

  Inherits property accessors from `RewriterStage` -> `RewriterProxy`.
  """

  # --- Shared Helpers Implementation ---

  def _cst_to_string(self, node: cst.BaseExpression) -> Optional[str]:
    """Flatten CST node to string."""
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute):
      base = self._cst_to_string(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    return None

  def _create_name_node(self, api_path: str) -> cst.BaseExpression:
    """Create a CST node for dotted path."""
    parts = api_path.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))
    return node

  def _create_dotted_name(self, name_str: str) -> cst.BaseExpression:
    """Alias for create_name_node."""
    return self._create_name_node(name_str)

  def _get_mapping(self, name: str, silent: bool = False) -> Optional[Dict[str, Any]]:
    """
    Query semantics for a mapping to the target framework.
    """
    lookup = self.semantics.get_definition(name)
    if not lookup:
      # Strict Mode Logic
      is_known_source_prefix = False
      root = name.split(".")[0]
      if root == self.source_fw or (self._alias_map and root in self._alias_map):
        is_known_source_prefix = True

      if self.strict_mode and is_known_source_prefix and not silent:
        self._report_failure(f"API '{name}' not found in semantics.")
      return None

    abstract_id, details = lookup

    if not self.semantics.is_verified(abstract_id):
      if not silent:
        self._report_failure(f"Skipped '{name}': Marked unsafe by verification report.")
      return None

    target_impl = self.semantics.resolve_variant(abstract_id, self.target_fw)

    if target_impl:
      get_tracer().log_match(
        source_api=name,
        target_api=target_impl.get("api", "Plugin Logic"),
        abstract_op=abstract_id,
      )
    else:
      if self.strict_mode and not silent:
        self._report_failure(f"No mapping available for '{name}' -> '{self.target_fw}'")
      return None

    return target_impl

  def _handle_variant_imports(self, variant: Dict[str, Any]) -> None:
    """Inject required imports into preamble."""
    reqs = variant.get("required_imports", [])
    for r in reqs:
      stmt = ""
      if isinstance(r, str):
        clean = r.strip()
        if clean.startswith("import") or clean.startswith("from"):
          stmt = clean
        else:
          stmt = f"import {clean}"
      elif isinstance(r, dict):
        mod = r.get("module")
        alias = r.get("alias")
        if mod:
          if alias:
            stmt = f"import {mod} as {alias}"
          else:
            stmt = f"import {mod}"

      if stmt:
        self.ctx.inject_preamble(stmt)


# Legacy Alias
CallMixin = ApiStage
