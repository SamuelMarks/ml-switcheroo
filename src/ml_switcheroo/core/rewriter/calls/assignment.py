"""
Assignment Unwrapping Logic.

This module provides the ``AssignmentUnwrapMixin``, which implements the logic for
handling assignment statements during the rewriting process. Specifically, it
focuses on unwrapping functional patterns (like ``apply((vars, x))``) into
cleaner object-oriented calls when targeting frameworks that support it.
"""

import libcst as cst

from ml_switcheroo.core.rewriter.calls.traits_cache import TraitsCachingMixin
from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.utils.node_diff import capture_node_source


class AssignmentUnwrapMixin(TraitsCachingMixin):
  """
  Mixin for transforming Assign nodes.

  Responsible for detecting functional assignments (e.g. ``y, state = layer.apply(...)``)
  and simplifying them based on the source framework's traits (e.g. converting to ``y = layer(...)``).
  """

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """
    Handles assignment unwrapping for Functional -> OOP transitions.

    Scenario: ``y, updates = layer.apply(vars, x)``
    Target:   ``y = layer(x)`` (NNX/Torch style)

    Reads ``functional_execution_method`` from Source Traits to detect pattern.

    Args:
        original_node: The node before transformation.
        updated_node: The node after transformation.

    Returns:
        The potentially unwrapped assignment node.
    """
    if not isinstance(original_node.value, cst.Call):
      return super().leave_Assign(original_node, updated_node)

    # Dynamic detection based on source trait (e.g. "apply", "call_fn")
    source_traits = self._get_source_traits()
    unwrap_method = source_traits.functional_execution_method

    if is_functional_apply(original_node.value, unwrap_method):
      if len(updated_node.targets) == 1:
        target = updated_node.targets[0].target
        if isinstance(target, (cst.Tuple, cst.List)):
          elements = target.elements
          if len(elements) > 0:
            primary_target = elements[0].value
            new_target = cst.AssignTarget(target=primary_target)

            new_node = updated_node.with_changes(targets=[new_target])
            get_tracer().log_mutation(
              "Assignment Unwrapping",
              capture_node_source(original_node),
              capture_node_source(new_node),
            )
            return new_node

    return super().leave_Assign(original_node, updated_node)
