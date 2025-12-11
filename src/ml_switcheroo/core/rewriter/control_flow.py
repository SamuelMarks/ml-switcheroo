"""
Control Flow Rewriting Logic.

Handles structural control flow transformations, specifically `for` loops.
Unlike API calls which map via Semantics JSON, control flow hooks are
dispatched via reserved system triggers (e.g. `transform_for_loop`).

This allows plugins to implement complex logic like:
- `for i in range(N)` -> `jax.lax.fori_loop(...)`
- `for x in iter`     -> `jax.lax.scan(...)`
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
    Invokes the 'transform_for_loop' hook if registered.

    If no plugin is registered or the plugin returns the node unmodified,
    the loop preserves its Python semantics (appropriate for Torch/NumPy).
    If JAX/TF is the target, a plugin is expected to intercept this.
    """
    hook = get_hook("transform_for_loop")

    if hook:
      # Hooks can return arbitrary CSTNodes (e.g. a Call replacement)
      try:
        new_node = hook(updated_node, self.ctx)
        # If transformation occurred, return it
        if new_node is not updated_node:
          return new_node
      except Exception as e:
        # If plugin crashes, fallback to warning mechanic defined in BaseRewriter
        self._report_failure(f"Loop transformation failed: {str(e)}")
        return original_node

    # Default: Pass through
    return updated_node
