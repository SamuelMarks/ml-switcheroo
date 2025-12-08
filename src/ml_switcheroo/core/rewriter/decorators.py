"""
Decorator Rewriting Mixin.

This module provides logic to handle Python decorators (e.g., `@torch.jit.script`).
It enables:
1.  **Renaming**: Mapping decorators between frameworks (e.g., `@torch.jit.script` -> `@jax.jit`).
2.  **Removal**: stripping decorators that have no equivalent in the target framework (if mapped to null).
"""

from typing import Union
import libcst as cst

from ml_switcheroo.core.rewriter.base import BaseRewriter


class DecoratorMixin(BaseRewriter):
  """
  Mixin for transforming Decorator nodes.
  """

  def leave_Decorator(
    self, original_node: cst.Decorator, updated_node: cst.Decorator
  ) -> Union[cst.Decorator, cst.RemovalSentinel]:
    """
    Processes decorators attached to functions or classes.

    Logic:
    1. Identifies the decorator name (handling both `@name` and `@call(...)`).
    2. Looks up the semantic definition.
    3. If the target variant is explicitly `null` (None), removes the decorator.
    4. If the target variant specifies a new API, rewrites the name.
    """
    # 1. Extract the name expression
    expr = updated_node.decorator
    func_node = None

    if isinstance(expr, cst.Call):
      func_node = expr.func
    else:
      func_node = expr

    name = self._get_qualified_name(func_node)
    if not name:
      return updated_node

    # 2. Lookup Semantics
    # We access the raw definition to distinguish between "Missing" and "Explicit None"
    lookup = self.semantics.get_definition(name)
    if not lookup:
      return updated_node

    _, details = lookup
    variants = details.get("variants", {})

    # Check if target framework is even considered in the spec for this op
    if self.target_fw not in variants:
      # If strict mode is on, we might want to flag this, but BaseRewriter
      # error reporting is usually per-statement line.
      # Decorators are part of the statement (FunctionDef/ClassDef).
      # We defer strict checking to the specific mixins or assume pass-through if not mapped.
      return updated_node

    target_variant = variants[self.target_fw]

    # Case A: Explicit Removal (mapped to null)
    if target_variant is None:
      return cst.RemoveFromParent()

    # Case B: Rewrite Name (API Mapping)
    # Note: If it was a Call (`@foo(...)`), CallMixin might have already rewritten
    # the inner call `foo(...)` -> `bar(...)`.
    # However, CallMixin relies on `_get_qualified_name` correctly resolving the function.
    # If `AttributeMixin` skipped rewriting the name because it looked like a function,
    # and `CallMixin` processed args but kept the func name (if it match logic?),
    # or if it's NOT a call (just `@foo`), we need to handle it here.

    target_api = target_variant.get("api")
    if target_api:
      # Check if we need to apply the rewrite
      current_name = self._get_qualified_name(func_node)
      if current_name != target_api:
        new_name_node = self._create_name_node(target_api)

        if isinstance(expr, cst.Call):
          # Update the function part of the Call
          new_expr = expr.with_changes(func=new_name_node)
        else:
          # Update the expression directly
          new_expr = new_name_node

        return updated_node.with_changes(decorator=new_expr)

    return updated_node
