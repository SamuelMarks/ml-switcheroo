"""
Control Flow Rewriting Logic.

Handles structural control flow transformations, specifically ``for`` loops.
Unlike API calls which map via Semantics JSON, control flow hooks are
dispatched via reserved system triggers (e.g. ``transform_for_loop``).

This logic enables a prioritization chain for loop handling:

1.  **Static Unrolling**: Optimization strategy. If a loop is determined to be
    static (e.g., ``range(5)``), it is unrolled into flat code.
2.  **Safety Transformation**: Compliance strategy. If the loop remains dynamic
    and the target framework (like JAX) enforces functional purity, this ensures
    the loop is flagged or converted to a scan/while construct.
"""

from typing import Union
import libcst as cst

from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.core.hooks import get_hook


class ControlFlowMixin(BaseRewriter):
  """
  Mixin for visiting Control Flow nodes (For, While, If).
  """

  def leave_For(self, original_node: cst.For, updated_node: cst.For) -> Union[cst.For, cst.CSTNode]:
    """
    Invokes loop transformation logic.

    Implements a priority chain:
    1.  **Static Unroll**: Checks ``transform_for_loop_static``. If the loop index
        is constant, unrolls it (Optimization).
    2.  **General Transform**: Checks ``transform_for_loop``. Handles general
        case logic or applies safety warnings (e.g., Escape Hatch for JAX).

    Args:
        original_node (cst.For): The node before transformation.
        updated_node (cst.For): The node after child visitors have run.

    Returns:
        Union[cst.For, cst.CSTNode]: The transformed node (potentially a
        FlattenSentinel of unrolled statements) or the original node if untouched.
    """
    # 1. Attempt Static Unroll Optimization (Wire-in for Feature 08)
    static_hook = get_hook("transform_for_loop_static")

    if static_hook:
      try:
        # Logic: If unroll hook returns 'updated_node', it refused to unroll.
        # If it returns anything else (e.g., FlattenSentinel), it worked.
        new_node = static_hook(updated_node, self.ctx)
        if new_node is not updated_node:
          return new_node
      except Exception as e:
        # Log failure but proceed to standard handler to ensure safety rules apply
        self._report_warning(f"Static loop unrolling failed: {str(e)}")

    # 2. General Loop Transformation / Safety Scanner
    hook = get_hook("transform_for_loop")

    if hook:
      # Hooks can return arbitrary CSTNodes
      try:
        new_node = hook(updated_node, self.ctx)
        # If transformation occurred, return it
        if new_node is not updated_node:
          return new_node
      except Exception as e:
        # If plugin crashes, fallback to warning mechanic defined in BaseRewriter
        self._report_failure(f"Loop transformation failed: {str(e)}")
        return original_node

    # Default: Pass through (e.g. for Torch targets which support python loops)
    return updated_node
