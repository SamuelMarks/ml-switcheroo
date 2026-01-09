"""
Auxiliary Stage: Decorator Handling Logic (Partial Definition).

This module defines the `AuxiliaryStage` class, which aggregates decorator
processing logic. It handles renaming or removing decorators based on semantic
mappings and target framework traits.

Note: This file provides the `DecoratorMixin` logic which is mixed into `AuxiliaryStage`
alongside `ControlFlowMixin` from `control_flow.py`.
"""

from typing import Union, Optional, TYPE_CHECKING
import libcst as cst
from ml_switcheroo.semantics.schema import StructuralTraits

if TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.control_flow import AuxiliaryStage


class DecoratorMixin:
  """
  Mixin for transforming Decorator nodes.

  Expects to be mixed into `AuxiliaryStage` which provides `context`.
  """

  _cached_traits: Optional[StructuralTraits] = None

  def _get_traits(self: "AuxiliaryStage") -> StructuralTraits:
    """Lazily loads structural traits for the current target framework."""
    if self._cached_traits:
      return self._cached_traits

    conf = self.context.semantics.get_framework_config(self.context.target_fw)
    if conf and "traits" in conf:
      self._cached_traits = StructuralTraits.model_validate(conf["traits"])
      return self._cached_traits  # type: ignore

    return StructuralTraits()

  def leave_Decorator(
    self: "AuxiliaryStage", original_node: cst.Decorator, updated_node: cst.Decorator
  ) -> Union[cst.Decorator, cst.RemovalSentinel]:
    """
    Processes decorators attached to functions or classes.

    Logic:
    1. Identifies the decorator name from `original_node` to ensure we key off the
       Source Framework API.
    2. Looks up the semantic definition.
    3. If the target variant is explicitly `null` (None), returns RemovalSentinel.
    4. If the target variant specifies a new API, rewrites the name.
    """
    # 1. Extract the name expression from ORIGINAL node for lookup stability
    expr = original_node.decorator
    func_node = None

    if isinstance(expr, cst.Call):
      func_node = expr.func
    else:
      func_node = expr

    # Use helper from BaseRewriter logic (via Stage mixin/context) to resolve alias
    name = self._get_qualified_name(func_node)
    if not name:
      return updated_node

    # 2. Lookup Semantics
    lookup = self.context.semantics.get_definition(name)
    if not lookup:
      return updated_node

    _, details = lookup
    variants = details.get("variants", {})

    # Check if target framework has a definition
    if self.context.target_fw not in variants:
      # If not mapped, preserve original
      return updated_node

    target_variant = variants[self.context.target_fw]

    # Case A: Explicit Removal (mapped to null)
    if target_variant is None:
      # FIXED: Use correct LibCST sentinel
      return cst.RemoveFromParent()

    # Case B: Rewrite Name (API Mapping)
    target_api = target_variant.get("api")

    if target_api:
      # Construct the new API node (e.g. jax.jit)
      new_name_node = self._create_dotted_name(target_api)
      current_expr = updated_node.decorator

      # Apply replacement logic based on syntactic structure
      if isinstance(current_expr, cst.Call):
        # The decorator has arguments: @foo(bar=1)
        # We simply swap the function name being called.
        new_expr = current_expr.with_changes(func=new_name_node)
      else:
        # The decorator is a simple reference: @foo
        # We replace strictly.
        new_expr = new_name_node

      return updated_node.with_changes(decorator=new_expr)

    return updated_node

  # --- Helpers expected from Aggregator --

  def _get_qualified_name(self, node: cst.BaseExpression) -> Optional[str]:
    """Proxy: implemented in context-aware base."""
    full_str = self.__cst_to_string(node)
    if not full_str:
      return None

    parts = full_str.split(".")
    root = parts[0]

    if root in self.context.alias_map:
      canonical_root = self.context.alias_map[root]
      if len(parts) > 1:
        return f"{canonical_root}.{'.'.join(parts[1:])}"
      return canonical_root

    return full_str

  def __cst_to_string(self, node: cst.BaseExpression) -> Optional[str]:
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.Attribute):
      base = self.__cst_to_string(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    return None

  def _create_dotted_name(self, name_str: str) -> cst.BaseExpression:
    parts = name_str.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))
    return node
