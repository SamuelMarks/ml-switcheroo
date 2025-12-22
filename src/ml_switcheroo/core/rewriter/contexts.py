"""
ContextRewriter Mixin.

This module provides the `ContextMixin` for the `PivotRewriter`,
handling transformations of `With` blocks (Context Managers).

Capabilities:
1.  **API Mapping**: Maps context managers (e.g., `torch.no_grad() -> contextlib.nullcontext()`).
2.  **Structural Stripping**: Supports completely removing the `with` block while preserving the
    body indentation (Lifting), if the semantic mapping dictates `transformation_type="strip_context"`.
"""

import libcst as cst
from typing import Union, List, Optional
from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.core.dsl import OpType


class ContextMixin(BaseRewriter):
  """
  Mixin for transforming `With` blocks.
  """

  def leave_With(self, original_node: cst.With, updated_node: cst.With) -> Union[cst.With, cst.FlattenSentinel]:
    """
    Processes 'with' statements.

    Logic:
    1. Iterate over `with` items (expressions).
    2. Identify if the expression corresponds to a `CONTEXT` OpType in semantics.
    3. Check transformation logic:
       - If `strip_context`, identifying marker is found, lift the body.
       - Otherwise, allow `CallMixin` (which runs on children) to have already renamed the API.
    """

    # We need to check if any of the items request a "strip_context" transformation.
    # Since CallMixin runs on children *before* leave_With, the item node inside updated_node
    # might already be renamed (e.g. to 'contextlib.nullcontext').
    # However, to decide on stripping, we need to trace back to the Semantic Definition.

    # We reconstruct the Abstract ID lookup from the *original* node to be safe,
    # or check the updated node if we trust the rewriter's cache.

    should_strip = False

    for item in original_node.items:
      expr = item.item
      if isinstance(expr, cst.Call):
        name = self._get_qualified_name(expr.func)
        if name:
          defn = self.semantics.get_definition(name)
          if defn:
            op_id, details = defn
            # Check Target Variant logic
            target_variant = details.get("variants", {}).get(self.target_fw, {})

            if target_variant and target_variant.get("transformation_type") == "strip_context":
              should_strip = True
              break

    if should_strip:
      # Return the body statements flattened into the parent scope (Lifting)
      # The body is a IndentedBlock. We need its children.
      body_block = updated_node.body
      if isinstance(body_block, cst.IndentedBlock):
        # FlattenSentinel takes a list of statements
        return cst.FlattenSentinel(body_block.body)

    return updated_node
