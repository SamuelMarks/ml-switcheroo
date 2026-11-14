"""Call Rewrite Collaborator for ``ApiTransformer``.

Provides ``CallRewriteHandler``, which orchestrates pre-checks, explicit and
implicit mapping resolution, strict-mode diagnostics, version checks, strategy
execution, and post-processing for ``cst.Call`` nodes.
"""

from __future__ import annotations

import typing

import libcst as cst

from ml_switcheroo.core.rewriter.calls.post import handle_post_processing
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method
from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy
from ml_switcheroo.core.rewriter.calls.utils import (
  is_builtin,
  is_neural_on_math_backend,
  is_super_call,
  log_diff,
)
from ml_switcheroo.core.tracer import get_tracer

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.interface import RewriterHost


@typing.runtime_checkable
class _ResolveOpIdProtocol(typing.Protocol):
  """Protocol for semantics objects providing ``resolve_op_id``."""

  def resolve_op_id(self, source_framework: str, func_name: str) -> str | None:
    """Resolve abstract operation ID from source framework and function name.

    Args:
        source_framework: Source framework key.
        func_name: Function name.

    Returns:
        Resolved operation ID or ``None``.
    """
    ...


class CallRewriteHandler:
  """Single-responsibility handler rewriting ``cst.Call`` expressions over a ``RewriterHost``.

  Attributes:
      host: Bound ``RewriterHost`` (e.g. ``ApiTransformer``) supplying state and resolution hooks.
  """

  def __init__(self, host: RewriterHost) -> None:
    """Initialize ``CallRewriteHandler`` bound to ``host``.

    Args:
        host: Bound ``RewriterHost`` instance.
    """
    self.host: RewriterHost = host

  def rewrite_call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
    """Rewrite a ``cst.Call`` node from the source framework to the target framework.

    Args:
        original_node: Original ``cst.Call`` node prior to child traversal.
        updated_node: Updated ``cst.Call`` node after child traversal.

    Returns:
        Rewritten ``cst.BaseExpression`` or ``updated_node`` when unmapped.
    """
    func_name = self.host._get_qualified_name(original_node.func)

    handled, result_node = handle_pre_checks(self.host, original_node, updated_node, func_name)
    if handled:
      return result_node

    mapping = self.host._get_mapping(func_name) if func_name else None

    if not mapping:
      guessed_name = resolve_implicit_method(self.host, original_node, func_name)
      if guessed_name:
        mapping = self.host._get_mapping(guessed_name, silent=True)
        func_name = guessed_name

    if not mapping:
      if is_super_call(original_node):
        return updated_node

      if func_name and not is_builtin(func_name):
        get_tracer().log_inspection(node_str=func_name, outcome="Skipped", detail="No Entry in Semantics Knowledge Base")

      if self.host.strict_mode and func_name and func_name.startswith(f"{self.host.source_framework}."):
        sem_obj = self.host.semantics
        abstract_id = (
          sem_obj.resolve_op_id(self.host.source_framework, func_name)
          if isinstance(sem_obj, _ResolveOpIdProtocol)
          else None
        )

        try:
          origins = self.host.semantics._key_origins
        except AttributeError:
          origins = {}

        lookup_id = abstract_id
        if not lookup_id and func_name:
          lookup_id = func_name.split(".")[-1]

        tier = origins.get(lookup_id) if isinstance(lookup_id, str) else None

        if is_neural_on_math_backend(tier, self.host.target_framework):
          self.host._report_failure(
            f"Cannot map neural network abstraction '{func_name}' directly to pure math backend "
            f"'{self.host.target_framework}'. Use a framework like Flax or Keras."
          )
        else:
          self.host._report_failure(f"API '{func_name}' not found in semantics.")

      return updated_node

    raw_min_v = mapping.get("min_version")
    raw_max_v = mapping.get("max_version")
    min_v = str(raw_min_v) if isinstance(raw_min_v, str) else None
    max_v = str(raw_max_v) if isinstance(raw_max_v, str) else None
    v_warn = self.host.check_version_constraints(min_v, max_v)
    if v_warn:
      self.host._report_warning(v_warn)

    lookup = self.host.semantics.get_definition(str(func_name or ""))
    if not lookup:
      return updated_node

    abstract_id, details = lookup

    result_node = execute_strategy(self.host, original_node, updated_node, mapping, details, abstract_id)
    result_node = handle_post_processing(self.host, result_node, mapping, abstract_id)

    log_diff(f"Operation ({abstract_id})", original_node, result_node)
    return result_node


__all__ = ["CallRewriteHandler"]
