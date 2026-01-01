"""
Post-processing Phase for Call Rewriting.

Handles transformations that occur *after* the main API rewrite call has been constructed.

1.  **Output Adaptation**: Applying `output_adapter` lambdas or `output_select_index` subscripting.
2.  **Output Casting**: Injecting `.astype(...)` based on `output_cast`.
3.  **State Threading (Init)**: Stripping or injecting magic arguments (like `rngs`)
    if inside a constructor context, based on Target Framework Traits.
"""

from typing import Any, Dict
import libcst as cst

from ml_switcheroo.core.rewriter.calls.transformers import (
  apply_index_select,
  apply_output_adapter,
)
from ml_switcheroo.core.rewriter.calls.utils import inject_kwarg, strip_kwarg
from ml_switcheroo.enums import SemanticTier


def handle_post_processing(
  rewriter: Any,
  node: cst.CSTNode,
  mapping: Dict[str, Any],
  abstract_id: str,
) -> cst.CSTNode:
  """
  Applies post-rewrite modifications to the result node.

  Args:
      rewriter: The calling Rewriter instance.
      node: The CST node resulting from the transformation strategy.
      mapping: The definition of the target variant.
      abstract_id: The ID of the abstract operation.

  Returns:
      cst.CSTNode: The final enhanced node.
  """
  result_node = node

  # 1. Output Adaptation (Tuple selection / formatting)
  if "output_select_index" in mapping and mapping["output_select_index"] is not None:
    try:
      result_node = apply_index_select(result_node, mapping["output_select_index"])
    except Exception as e:
      rewriter._report_failure(f"Output indexing failed: {e}")
      return result_node
  elif "output_adapter" in mapping and mapping["output_adapter"]:
    try:
      result_node = apply_output_adapter(result_node, mapping["output_adapter"])
    except Exception as e:
      rewriter._report_failure(f"Output adapter failed: {e}")
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
      # Ignore if AST construction fails (rare)
      pass

  # 3. State Threading in Constructors (Logic 4)
  if (
    rewriter._signature_stack and rewriter._signature_stack[-1].is_init and rewriter._signature_stack[-1].is_module_method
  ):
    # Determine if this op corresponds to a Neural Component (Layer/Module)
    origins = getattr(rewriter.semantics, "_key_origins", {})
    tier = origins.get(abstract_id)
    traits = rewriter._get_target_traits()
    is_neural = tier == SemanticTier.NEURAL.value

    force = False
    # Check if we should force injection based on magic args presence in current call
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
        # A. Inject Magic Arguments (Target requirement)
        for arg_name, _ in traits.inject_magic_args:
          result_node = inject_kwarg(result_node, arg_name, arg_name)

        # B. Strip Magic Arguments (Source artifact removal)
        args_to_strip = set(traits.strip_magic_args)
        if traits.auto_strip_magic_args and hasattr(rewriter.semantics, "known_magic_args"):
          args_to_strip.update(rewriter.semantics.known_magic_args)
          # Don't strip what we just injected (native args)
          native = {a[0] for a in traits.inject_magic_args}
          args_to_strip -= native

        for arg_name in args_to_strip:
          result_node = strip_kwarg(result_node, arg_name)

  return result_node
