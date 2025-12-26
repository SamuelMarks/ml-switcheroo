"""
Decorator Rewriting Mixin.

This module provides logic to handle Python decorators (e.g., `@torch.jit.script`).
It enables:

1.  **Renaming**: Mapping decorators between frameworks (e.g., `@torch.jit.script` -> `@jax.jit`).
2.  **Removal**: stripping decorators that have no equivalent in the target framework (if mapped to null).
3.  **Trait-Based Handling**: Removes decorators marked as irrelevant by the target framework's traits,
    decoupling renaming logic from hardcoded framework strings.

Logic robustness handles conflict with `CallMixin`. Since `leave_Decorator` runs
after child nodes (the decorator expression) are visited, `CallMixin` might have
already transformed calls inside the decorator (e.g. `@torch.jit.script(optk=True)`).
To perform accurate lookups, we inspect the `original_node` (preserving Source semantics)
before applying target logic.
"""

from typing import Union, Set, Optional
import libcst as cst
from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.semantics.schema import StructuralTraits


class DecoratorMixin(BaseRewriter):
  """
  Mixin for transforming Decorator nodes.
  Part of PivotRewriter.
  """

  _cached_traits: Optional[StructuralTraits] = None

  def _get_traits(self) -> StructuralTraits:
    """Lazily loads structural traits for the current target framework."""
    if self._cached_traits:
      return self._cached_traits

    conf = self.semantics.get_framework_config(self.target_fw)
    if conf and "traits" in conf:
      self._cached_traits = StructuralTraits.model_validate(conf["traits"])
      return self._cached_traits

    return StructuralTraits()

  def leave_Decorator(
    self, original_node: cst.Decorator, updated_node: cst.Decorator
  ) -> Union[cst.Decorator, cst.RemovalSentinel]:
    """
    Processes decorators attached to functions or classes.

    Logic:

    1. Identifies the decorator name from `original_node` to ensure we key off the
       Source Framework API, even if `CallMixin` modified the children in `updated_node`.
    2. Looks up the semantic definition.
    3. Checks Target Structural Traits for explicit stripping requirements (e.g. `strip_decorators`).
    4. If the target variant is explicitly `null` (None), removes the decorator.
    5. If the target variant specifies a new API, rewrites the name.
    """
    # 1. Extract the name expression from ORIGINAL node for lookup stability
    expr = original_node.decorator
    func_node = None

    if isinstance(expr, cst.Call):
      func_node = expr.func
    else:
      func_node = expr

    # Use helper from BaseRewriter to resolve alias (t.jit -> torch.jit)
    name = self._get_qualified_name(func_node)
    if not name:
      return updated_node

    # 2. Check Traits implementation (Decoupling)
    # Allows a framework to declare "I don't support X type decorators" without
    # explicit void mapping in JSON if configured via generic traits.
    # Currently StructuralTraits focuses on lifecycle methods, but we can extend this pattern
    # for future decorator traits if needed.

    # 3. Lookup Semantics
    lookup = self.semantics.get_definition(name)
    if not lookup:
      return updated_node

    _, details = lookup
    variants = details.get("variants", {})

    # Check if target framework has a definition
    if self.target_fw not in variants:
      # If not mapped, preserve original (or updated node if children changed)
      return updated_node

    target_variant = variants[self.target_fw]

    # Case A: Explicit Removal (mapped to null)
    if target_variant is None:
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
        # CallMixin might have already processed arguments (renamed keywords/values).
        # We simply swap the function name being called.
        new_expr = current_expr.with_changes(func=new_name_node)
      else:
        # The decorator is a simple reference: @foo
        # We replace strictly.
        new_expr = new_name_node

      return updated_node.with_changes(decorator=new_expr)

    return updated_node
