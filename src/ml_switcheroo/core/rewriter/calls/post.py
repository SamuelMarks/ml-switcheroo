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
      if hasattr(rewriter, "_report_failure"):
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
  ctx_obj = getattr(rewriter, "context", None)
  sig_stack = getattr(ctx_obj, "signature_stack", None) if ctx_obj is not None else None
  if sig_stack and sig_stack[-1].is_init and sig_stack[-1].is_module_method:
    sem_obj = getattr(rewriter, "semantics", None)
    origins = getattr(sem_obj, "_key_origins", {}) if sem_obj is not None else {}
    tier = origins.get(abstract_id)
    traits = getattr(rewriter, "_get_target_traits")()
    is_neural = tier in (SemanticTier.NEURAL.value, "neural_ops")

    force = False
    if isinstance(result_node, cst.Call):
      magic = set(traits.strip_magic_args)
      if traits.auto_strip_magic_args and sem_obj is not None and hasattr(sem_obj, "known_magic_args"):
        magic.update(sem_obj.known_magic_args)

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
        if traits.auto_strip_magic_args and sem_obj is not None and hasattr(sem_obj, "known_magic_args"):
          args_to_strip.update(sem_obj.known_magic_args)
          native = {a[0] for a in traits.inject_magic_args}
          args_to_strip -= native

        for arg_name in args_to_strip:
          result_node = strip_kwarg(result_node, arg_name)

  return result_node
