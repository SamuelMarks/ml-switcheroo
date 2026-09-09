"""Post-processing Phase for Call Rewriting.

Handles output adaptation and state threading.
Updated to prevent imports from legacy modules and removed deprecated output adapter hooks.
"""

from typing import TYPE_CHECKING, List, Tuple
import libcst as cst

from ml_switcheroo.core.rewriter.calls.transformers import apply_index_select
from ml_switcheroo.core.rewriter.calls.utils import inject_kwarg, strip_kwarg
from ml_switcheroo_ir.schema.ghost import SemanticTier

if TYPE_CHECKING:
  # Structural typing for rewriter to avoid circular import
  class SemanticManagerDummy:
    """Structural type representation of a semantic manager."""

    known_magic_args: List[str]

  class TargetTraitsDummy:
    """Structural type representation of a semantic manager."""

    strip_magic_args: List[str]
    auto_strip_magic_args: bool
    inject_magic_args: List[Tuple[str, str]]

  class SignatureStackDummy:
    """Structural type representation of a semantic manager."""

    is_init: bool
    is_module_method: bool

  class RewriterContextDummy:
    """Structural type representation of a semantic manager."""

    signature_stack: List[SignatureStackDummy]

  class RewriterDummy:
    """Structural type representation of a semantic manager."""

    semantics: SemanticManagerDummy
    context: RewriterContextDummy

    def _report_failure(self, msg: str) -> None:
      """Structural method signature."""
      ...

    def _create_dotted_name(self, name: str) -> cst.Attribute:
      """Structural method signature."""
      ...

    def _get_target_traits(self) -> TargetTraitsDummy:
      """Structural method signature."""
      ...


def handle_post_processing(
  rewriter: "RewriterDummy",
  node: cst.BaseExpression,
  mapping: dict,
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
  if "output_select_index" in mapping and mapping["output_select_index"] is not None:
    try:
      result_node = apply_index_select(result_node, mapping["output_select_index"])
    except Exception as e:
      if hasattr(rewriter, "_report_failure"):
        rewriter._report_failure(f"Output indexing failed: {e}")
      return result_node

  # 2. Output Casting
  if "output_cast" in mapping and mapping["output_cast"]:
    try:
      type_node = rewriter._create_dotted_name(mapping["output_cast"])
      result_node = cst.Call(
        func=cst.Attribute(value=result_node, attr=cst.Name("astype")),
        args=[cst.Arg(value=type_node)],
      )
    except Exception:
      pass

  # 3. State Threading in Constructors (Logic 4)
  # Check signature stack
  if (
    hasattr(rewriter.context, "signature_stack")
    and rewriter.context.signature_stack
    and rewriter.context.signature_stack[-1].is_init
    and rewriter.context.signature_stack[-1].is_module_method
  ):
    origins = getattr(rewriter.semantics, "_key_origins", {})
    tier = origins.get(abstract_id)
    traits = rewriter._get_target_traits()
    is_neural = tier in (SemanticTier.NEURAL.value, "neural_ops")

    force = False
    if isinstance(result_node, cst.Call):
      magic = set(traits.strip_magic_args)
      if traits.auto_strip_magic_args and hasattr(rewriter.semantics, "known_magic_args"):
        magic.update(rewriter.semantics.known_magic_args)

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
        if traits.auto_strip_magic_args and hasattr(rewriter.semantics, "known_magic_args"):
          args_to_strip.update(rewriter.semantics.known_magic_args)
          native = {a[0] for a in traits.inject_magic_args}
          args_to_strip -= native

        for arg_name in args_to_strip:
          result_node = strip_kwarg(result_node, arg_name)

  return result_node
