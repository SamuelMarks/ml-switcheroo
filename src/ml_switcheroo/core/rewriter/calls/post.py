"""Post-processing Phase for Call Rewriting.

Handles output adaptation and state threading.
Updated to prevent imports from legacy modules and removed deprecated output adapter hooks.
"""

from ml_switcheroo.enums import RuntimeTensorOrValue
import typing

import libcst as cst
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.core.rewriter.calls.transformers import apply_index_select
from ml_switcheroo.core.rewriter.calls.utils import inject_kwarg, strip_kwarg

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.interface import RewriterHost
  from ml_switcheroo.semantics.schema import StructuralTraits


@typing.runtime_checkable
class _FailureReporterProtocol(typing.Protocol):
  """Protocol for rewriters providing `_report_failure`."""

  def _report_failure(self, msg: str) -> None:
    """Report a failure message.

    Args:
        msg: Failure message.
    """
    ...


@typing.runtime_checkable
class _TargetTraitsGetterProtocol(typing.Protocol):
  """Protocol for rewriters providing `_get_target_traits`."""

  def _get_target_traits(self) -> "StructuralTraits":
    """Return target framework `StructuralTraits`.

    Returns:
        Resolved `StructuralTraits`.
    """
    ...


class PostRewriterProtocol(typing.Protocol):
  """Minimal rewriter protocol required by handle_post_processing."""

  def _create_dotted_name(self, name_str: str) -> cst.BaseExpression:
    """Create a dotted name expression from string.

    Args:
        name_str: Dotted name string.

    Returns:
        Created CST expression.
    """
    ...


def handle_post_processing(
  rewriter: "RewriterHost | PostRewriterProtocol",
  node: cst.BaseExpression,
  mapping: typing.Mapping[str, RuntimeTensorOrValue],
  abstract_id: str,
) -> cst.BaseExpression:
  """Apply post-rewrite modifications to the result node, such as type casting or state threading.

  Args:
      rewriter: The CST rewriter instance containing context.
      node: The node being processed.
      mapping: The configuration mapping for the rewrite.
      abstract_id: The abstract ID of the operation.

  Returns:
      cst.BaseExpression: The modified result node.

  """
  result_node = node

  # 1. Output Adaptation (Tuple selection / formatting)
  # Note: Legacy 'output_adapter' string logic has been deprecated and removed.
  # We only support structured 'output_select_index' for tuple destructuring.
  raw_select = mapping.get("output_select_index")
  if raw_select is not None:
    try:
      sel_idx = int(str(raw_select))
      result_node = apply_index_select(result_node, sel_idx)
    except Exception as e:
      if isinstance(rewriter, _FailureReporterProtocol):
        rewriter._report_failure(f"Output indexing failed: {e}")
      return result_node

  # 2. Output Casting
  raw_cast = mapping.get("output_cast")
  if isinstance(raw_cast, str) and raw_cast:
    try:
      type_node = rewriter._create_dotted_name(raw_cast)
      result_node = cst.Call(
        func=cst.Attribute(value=result_node, attr=cst.Name("astype")),
        args=[cst.Arg(value=type_node)],
      )
    except Exception:
      pass

  # 3. State Threading in Constructors (Logic 4)
  # Check signature stack
  from ml_switcheroo.core.rewriter.calls.pre import _resolve_attr

  ctx_obj = _resolve_attr(rewriter, "context")
  sig_stack = _resolve_attr(ctx_obj, "signature_stack")
  if isinstance(sig_stack, list) and len(sig_stack) > 0 and sig_stack[-1].is_init and sig_stack[-1].is_module_method:
    sem_obj = _resolve_attr(rewriter, "semantics")
    origins = _resolve_attr(sem_obj, "_key_origins")
    tier = origins.get(abstract_id) if isinstance(origins, dict) else None
    assert isinstance(rewriter, _TargetTraitsGetterProtocol)
    traits = rewriter._get_target_traits()
    is_neural = tier in (SemanticTier.NEURAL.value, "neural_ops")

    known_magic = _resolve_attr(sem_obj, "known_magic_args")
    force = False
    if isinstance(result_node, cst.Call):
      magic = set(traits.strip_magic_args)
      if traits.auto_strip_magic_args and isinstance(known_magic, (set, list, tuple)):
        magic.update(known_magic)

      for arg in result_node.args:
        if arg.keyword and arg.keyword.value in magic:
          force = True
          break

    if is_neural or force:
      if isinstance(result_node, cst.Call):
        # A. Inject Magic Arguments
        for arg_name, _ in traits.inject_magic_args:
          result_node = inject_kwarg(result_node, arg_name, arg_name)

        # B. Strip Magic Arguments
        args_to_strip = set(traits.strip_magic_args)
        if traits.auto_strip_magic_args and isinstance(known_magic, (set, list, tuple)):
          args_to_strip.update(known_magic)
          native = {a[0] for a in traits.inject_magic_args}
          args_to_strip -= native

        for arg_name in args_to_strip:
          result_node = strip_kwarg(result_node, arg_name)

  return result_node
