"""
CallMixin Definition (Aggregator).

Combines various mixins required to handle assignment operations and function
invocations in the AST.

It pulls together logic from:
- :class:`InvocationMixin`: Helper logic for complex Call handling.
- :class:`AssignmentUnwrapMixin`: Logic for unwrapping functional return tuples.
- :class:`BaseRewriter`: Core traversal state management.

Note: NormalizationMixin and TraitsCachingMixin are inherited implicitly via InvocationMixin.
"""

from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.core.rewriter.calls.assignment import AssignmentUnwrapMixin
from ml_switcheroo.core.rewriter.calls.invocation import InvocationMixin


class CallMixin(InvocationMixin, AssignmentUnwrapMixin, BaseRewriter):
  """
  Composite mixin for transforming Call nodes (`func(...)`) and Assignments (`x = ...`).

  This class aggregates:
  1.  **Functional Unwrapping**: Handling legacy `apply` patterns (via `AssignmentUnwrapMixin`).
  2.  **Call Rewriting**: The core engine for remapping APIs, handling plugins, and injecting state (via `InvocationMixin`).
  3.  **Trait caching**: Performance optimization for framework config lookup (Implicit).

  Inheritance Order is important: `InvocationMixin` and `AssignmentUnwrapMixin`
  override specific visitor methods, while `BaseRewriter` provides utilities.
  """
