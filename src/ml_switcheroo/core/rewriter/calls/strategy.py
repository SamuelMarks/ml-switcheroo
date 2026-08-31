"""Core Transformation Strategies for Call Rewriting."""

from typing import TYPE_CHECKING
import libcst as cst

from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.core.rewriter.calls.dispatch import evaluate_dispatch_rules
from ml_switcheroo.core.rewriter.calls.transformers import (
  rewrite_as_infix,
  rewrite_as_inline_lambda,
  rewrite_as_macro,
)
from ml_switcheroo.core.rewriter.calls.utils import (
  compute_permutation,
  inject_permute_call,
)
from ml_switcheroo.core.rewriter.calls.guards import apply_strict_guards
from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

if TYPE_CHECKING:
  # Structural typing for rewriter to avoid circular import
  class HookContextDummy:
    """Dummy hook context."""

    current_op_id: str

  class Any:
    """Dummy rewriter context."""

    hook_context: HookContextDummy

  from ml_switcheroo.semantics.manager import SemanticsManager

  class RewriterDummy:
    """Dummy rewriter."""

    context: Any
    source_fw: str
    target_fw: str
    strict_mode: bool
    semantics: SemanticsManager

    def _handle_variant_imports(self, mapping: dict) -> None:
      """Dummy."""
      ...

    def _is_module_alias(self, name: cst.BaseExpression) -> bool:
      """Dummy."""
      ...

    def _report_failure(self, msg: str) -> None:
      """Dummy."""
      ...

    def _create_name_node(self, name: str) -> cst.Name:
      """Dummy."""
      ...


def execute_strategy(
  rewriter: "RewriterDummy",
  original: cst.Call,
  updated: cst.Call,
  mapping: dict,
  details: dict,
  abstract_id: str,
) -> cst.BaseExpression:
  """Apply the appropriate transformation strategy for rewriting a function call.

  This function determines and executes the correct transformation mechanism
  (e.g., infix operators, inline lambdas, macro expansions, plugins, or standard
  API calls) based on the provided mapping rules and API details. It also handles
  dispatch rules, argument normalization, strict mode guards, and layout permutations.

  Args:
      rewriter: The active rewriter instance providing context, semantics, and target framework details.
      original: The original `cst.Call` node from the source tree before any modifications.
      updated: The currently updated `cst.Call` node, which may have already undergone pre-processing.
      mapping: The target framework-specific mapping dictionary detailing how to transform the call.
      details: The general API details dictionary for the abstract operation being mapped.
      abstract_id: The unique identifier for the abstract operation (e.g., 'math.add').

  Returns:
      cst.BaseExpression: The transformed CST node representing the rewritten call or expression.

  """
  if hasattr(rewriter.context, "hook_context"):
    rewriter.context.hook_context.current_op_id = abstract_id

  # Handle imports from ApiPass helper
  if hasattr(rewriter, "_handle_variant_imports"):
    rewriter._handle_variant_imports(mapping)

  # 1. Dispatch Rules
  if "dispatch_rules" in mapping and mapping["dispatch_rules"]:
    dispatched_api = evaluate_dispatch_rules(rewriter, original, mapping["dispatch_rules"], details)
    if dispatched_api:
      mapping = mapping.copy()
      mapping["api"] = dispatched_api

  trans_type = mapping.get("transformation_type")

  # 2. Infix
  if trans_type == "infix":
    try:
      norm_args = normalize_arguments(original, updated, details, mapping, rewriter.source_fw, rewriter._is_module_alias)
      return rewrite_as_infix(
        original,
        norm_args,
        mapping.get("operator"),  # type: ignore
        details.get("std_args", []),
      )
    except (ValueError, IndexError) as e:
      rewriter._report_failure(f"Infix/Prefix transformation failed: {e}")
      return updated

  # 3. Inline Lambda
  elif trans_type == "inline_lambda":
    try:
      norm_args = normalize_arguments(original, updated, details, mapping, rewriter.source_fw, rewriter._is_module_alias)
      return rewrite_as_inline_lambda(mapping["api"], norm_args)
    except Exception as e:
      rewriter._report_failure(f"Inline lambda transformation failed: {e}")
      return updated

  # 4. Plugin
  elif "requires_plugin" in mapping:
    plugin_name = mapping["requires_plugin"]
    hook = get_hook(plugin_name)
    if hook:
      return hook(updated, rewriter.context.hook_context)
    else:
      rewriter._report_failure(f"Missing required plugin: '{plugin_name}'")
      return updated

  # 5. Macro
  elif mapping.get("macro_template"):
    try:
      norm_args = normalize_arguments(original, updated, details, mapping, rewriter.source_fw, rewriter._is_module_alias)
      std_arg_names = []
      for item in details.get("std_args", []):
        if isinstance(item, (list, tuple)):
          std_arg_names.append(item[0])
        elif isinstance(item, dict):
          std_arg_names.append(item["name"])
        else:
          std_arg_names.append(item)
      return rewrite_as_macro(mapping["macro_template"], norm_args, std_arg_names)
    except Exception as e:
      rewriter._report_failure(f"Macro expansion failed: {e}")
      return updated

  # 6. Standard
  else:
    target_api = mapping.get("api")
    if not target_api:
      origins = getattr(rewriter.semantics, "_key_origins", {})
      tier = origins.get(abstract_id)

      if tier in ("neural", "neural_ops") and rewriter.target_fw in ("numpy", "jax"):
        msg = f"Cannot map neural network abstraction '{abstract_id}' directly to pure math backend '{rewriter.target_fw}'. Use a framework like Flax or Keras."
      else:
        msg = mapping.get(
          "missing_message",
          f"No mapping available for '{abstract_id}' -> '{rewriter.target_fw}'",
        )
      rewriter._report_failure(msg)
      return updated

    try:
      norm_args = normalize_arguments(original, updated, details, mapping, rewriter.source_fw, rewriter._is_module_alias)

      # Apply Strict Guards (Rank Checking)
      if rewriter.strict_mode:
        norm_args = apply_strict_guards(rewriter, norm_args, details, mapping)  # type: ignore

      new_func = rewriter._create_name_node(target_api)
      result_node = updated.with_changes(func=new_func, args=norm_args)

      # Layout Permutation Logic
      if "layout_map" in mapping and mapping["layout_map"]:
        result_node = _apply_layout_permutation(result_node, mapping, details, rewriter)

      return result_node

    except ValueError:
      rewriter._report_failure("Argument normalization failed")
      return updated


def _apply_layout_permutation(
  node: cst.Call,
  mapping: dict,
  details: dict,
  rewriter: "RewriterDummy",
) -> cst.Call:
  """Apply layout permutation to the arguments or return value of a call.

  This function modifies the arguments of a function call, or wraps the entire call,
  with permutation logic (e.g., transposing dimensions) based on the `layout_map`
  provided in the transformation mapping. This is commonly used when transforming
  between frameworks with different data layout conventions (like NCHW vs. NHWC).

  Args:
      node: The call node whose arguments or return value need layout permutation.
      mapping: The transformation mapping dictionary containing the 'layout_map' rules.
      details: The general API details dictionary specifying standard argument names.
      rewriter: The active rewriter instance providing semantics and target framework details.

  Returns:
      cst.Call: The modified call node with permuted arguments or a permutation wrapped around the return value.

  """
  layout_map = mapping["layout_map"]
  std_args_raw = details.get("std_args", [])
  idx = 0
  modified_args = list(node.args)

  for item in std_args_raw:
    arg_name = item.get("name") if isinstance(item, dict) else (item[0] if isinstance(item, (list, tuple)) else item)
    if arg_name and arg_name in layout_map:
      rule = layout_map[arg_name]
      if "->" in rule:
        src_l, tgt_l = rule.split("->")
        perm_indices = compute_permutation(src_l.strip(), tgt_l.strip())
        if perm_indices and idx < len(modified_args):
          original_arg = modified_args[idx]
          wrapped_val = inject_permute_call(
            original_arg.value,
            perm_indices,
            rewriter.semantics,
            rewriter.target_fw,
          )
          modified_args[idx] = original_arg.with_changes(value=wrapped_val)
    idx += 1

  node = node.with_changes(args=modified_args)

  if "return" in layout_map:
    rule = layout_map["return"]
    if "->" in rule:
      src_l, tgt_l = rule.split("->")
      perm_indices = compute_permutation(src_l.strip(), tgt_l.strip())
      if perm_indices:
        node = inject_permute_call(node, perm_indices, rewriter.semantics, rewriter.target_fw)

  return node
