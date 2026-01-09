"""
Auxiliary Stage: Control Flow Handling.

This module completes the `AuxiliaryStage` definition by adding loop transformation
logic. It aggregates `DecoratorMixin` and `ControlFlowMixin` into the final
`AuxiliaryStage` class.
"""

from typing import Union, TYPE_CHECKING
import libcst as cst

from ml_switcheroo.core.rewriter.base import RewriterStage
from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.core.escape_hatch import EscapeHatch

# Import Decorator Logic to aggregate
from ml_switcheroo.core.rewriter.decorators import DecoratorMixin


class ControlFlowMixin:
  """
  Mixin for visiting Control Flow nodes (For, While, If).
  Expects to be mixed into `AuxiliaryStage` (which inherits RewriterProxy properties).
  """

  def leave_For(self: "AuxiliaryStage", original_node: cst.For, updated_node: cst.For) -> Union[cst.For, cst.CSTNode]:
    """
    Invokes loop transformation logic.
    """
    # 1. Attempt Static Unroll Optimization
    static_hook = get_hook("transform_for_loop_static")

    if static_hook:
      try:
        # Use context hook proxy
        # Logic relies on self.context from RewriterProxy
        new_node = static_hook(updated_node, self.context.hook_context)
        if new_node is not updated_node:
          return new_node
      except Exception as e:
        # Log warning but proceed
        self._report_warning(f"Static loop unrolling failed: {str(e)}")

    # 2. General Loop Transformation / Safety Scanner
    hook = get_hook("transform_for_loop")

    if hook:
      try:
        new_node = hook(updated_node, self.context.hook_context)
        if new_node is not updated_node:
          return new_node
      except Exception as e:
        self._report_failure(f"Loop transformation failed: {str(e)}")
        return original_node

    return updated_node

  def _report_failure(self: "AuxiliaryStage", reason: str) -> None:
    """Propagate failure to context error list."""
    self._current_stmt_errors.append(reason)

  def _report_warning(self: "AuxiliaryStage", reason: str) -> None:
    """Propagate warning to context list."""
    self._current_stmt_warnings.append(reason)


class AuxiliaryStage(DecoratorMixin, ControlFlowMixin, RewriterStage):
  """
  Combined Transformer Stage for Auxiliary Logic (Decorators + Control Flow).

  This class is instantiated by the `ASTEngine` during the pipeline Execution.
  It operates on the shared `RewriterContext`.
  """

  pass
