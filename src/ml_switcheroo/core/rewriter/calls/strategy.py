"""
Core Transformation Strategies for Call Rewriting.

This module implements the logic for applying specific rewrite strategies
(Infix, Lambda, Plugin, Macro, Standard) once a mapping has been identified.
"""

from typing import Any, Dict
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


def execute_strategy(
  rewriter: Any,
  original: cst.Call,
  updated: cst.Call,
  mapping: Dict[str, Any],
  details: Dict[str, Any],
  abstract_id: str,
) -> cst.CSTNode:
  """
  Applies the appropriate transformation strategy based on the mapping configuration.

  Flow:
  1.  **Imports**: Injects required imports.
  2.  **Dispatch**: Evaluates conditional rules to refine the mapping.
  3.  **Strategy**: Dispatches to Infix, Lambda, Plugin, Macro, or Standard rewrite logic.

  Args:
      rewriter: The calling Rewriter instance.
      original: Valid CST Call node (Source).
      updated: CST Call node with transformed children.
      mapping: The target variant definition.
      details: The abstract operation details.
      abstract_id: The ID of the operation.

  Returns:
      cst.CSTNode: The transformed node.
  """
  rewriter.ctx.current_op_id = abstract_id
  rewriter._handle_variant_imports(mapping)

  # 1. Dispatch Rules (Runtime API Switching)
  if "dispatch_rules" in mapping and mapping["dispatch_rules"]:
    dispatched_api = evaluate_dispatch_rules(rewriter, original, mapping["dispatch_rules"], details)
    if dispatched_api:
      # Create local copy to avoid mutating cache
      mapping = mapping.copy()
      mapping["api"] = dispatched_api

  trans_type = mapping.get("transformation_type")

  # 2. Infix / Prefix Operator (e.g. add(a, b) -> a + b)
  if trans_type == "infix":
    try:
      norm_args = rewriter._normalize_arguments(original, updated, details, mapping)
      return rewrite_as_infix(
        original,
        norm_args,
        mapping.get("operator"),
        details.get("std_args", []),
      )
    except (ValueError, IndexError) as e:
      rewriter._report_failure(f"Infix/Prefix transformation failed: {e}")
      return updated

  # 3. Inline Lambda (e.g. gelu(x) -> (lambda x: ...)(x))
  elif trans_type == "inline_lambda":
    try:
      norm_args = rewriter._normalize_arguments(original, updated, details, mapping)
      return rewrite_as_inline_lambda(mapping["api"], norm_args)
    except Exception as e:
      rewriter._report_failure(f"Inline lambda transformation failed: {e}")
      return updated

  # 4. Plugin Delegation
  elif "requires_plugin" in mapping:
    plugin_name = mapping["requires_plugin"]
    hook = get_hook(plugin_name)
    if hook:
      # Let plugin handle everything
      return hook(updated, rewriter.ctx)
    else:
      rewriter._report_failure(f"Missing required plugin: '{plugin_name}'")
      return updated

  # 5. Macro Expansion
  elif mapping.get("macro_template"):
    try:
      norm_args = rewriter._normalize_arguments(original, updated, details, mapping)
      # Extract standard arg names for template matching
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

  # 6. Standard Rewrite (API Rename + Arg Normalization)
  else:
    target_api = mapping.get("api")
    if not target_api:
      msg = mapping.get(
        "missing_message",
        f"No mapping available for '{abstract_id}' -> '{rewriter.target_fw}'",
      )
      rewriter._report_failure(msg)
      return updated

    try:
      norm_args = rewriter._normalize_arguments(original, updated, details, mapping)

      # Apply Strict Guards (Rank Checking)
      if rewriter.strict_mode:
        norm_args = apply_strict_guards(rewriter, norm_args, details, mapping)

      new_func = rewriter._create_name_node(target_api)
      result_node = updated.with_changes(func=new_func, args=norm_args)

      # Layout Permutation Logic (Feature 5)
      # injects transpose calls on inputs or output
      if "layout_map" in mapping and mapping["layout_map"]:
        result_node = _apply_layout_permutation(result_node, mapping, details, rewriter)

      return result_node

    except ValueError:
      rewriter._report_failure("Argument normalization failed")
      return updated


def _apply_layout_permutation(
  node: cst.Call,
  mapping: Dict[str, Any],
  details: Dict[str, Any],
  rewriter: Any,
) -> cst.Call:
  """Helper to apply input/output tensor permutation."""
  layout_map = mapping["layout_map"]
  std_args_raw = details.get("std_args", [])
  idx = 0
  modified_args = list(node.args)

  # Input Permutation
  for item in std_args_raw:
    arg_name = item.get("name") if isinstance(item, dict) else (item[0] if isinstance(item, (list, tuple)) else item)
    if arg_name and arg_name in layout_map:
      rule = layout_map[arg_name]
      if "->" in rule:
        src_l, tgt_l = rule.split("->")
        perm_indices = compute_permutation(src_l.strip(), tgt_l.strip())
        if perm_indices and idx < len(modified_args):
          original_arg = modified_args[idx]
          # Wrap the argument value in a permute call
          wrapped_val = inject_permute_call(
            original_arg.value,
            perm_indices,
            rewriter.semantics,
            rewriter.target_fw,
          )
          modified_args[idx] = original_arg.with_changes(value=wrapped_val)
    idx += 1

  node = node.with_changes(args=modified_args)

  # Output Permutation
  if "return" in layout_map:
    rule = layout_map["return"]
    if "->" in rule:
      src_l, tgt_l = rule.split("->")
      perm_indices = compute_permutation(src_l.strip(), tgt_l.strip())
      if perm_indices:
        # Wrap the entire call in a permute call
        node = inject_permute_call(node, perm_indices, rewriter.semantics, rewriter.target_fw)

  return node
