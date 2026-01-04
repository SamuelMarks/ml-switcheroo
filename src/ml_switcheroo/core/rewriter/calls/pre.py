"""
Pre-processing Phase for Call Rewriting.

Handles logic that must occur before the core semantic lookup:

1.  **Functional Unwrapping**: Converting ``apply`` patterns if converting from functional frameworks.
2.  **Plugin Claims**: Heuristic execution of plugins (like in-place unrolling).
3.  **Lifecycle Management**: Stripping methods like ``.cuda()`` or ``.eval()``.
4.  **Stateful Calls**: Rewriting calls that require state injection.
5.  **Implicit Resolution**: Guessing API paths for method calls on objects, using
    Symbol Table inference or heuristic fallbacks.
"""

from typing import Tuple, Any, Optional

import libcst as cst

from ml_switcheroo.core.rewriter.calls.utils import (
  is_functional_apply,
  log_diff,
  rewrite_stateful_call,
)
from ml_switcheroo.core.hooks import get_hook


def handle_pre_checks(
  rewriter: Any, original: cst.Call, updated: cst.Call, func_name: Optional[str]
) -> Tuple[bool, cst.CSTNode]:
  """
  Executes pre-lookup checks and transformations.

  These checks handle framework-specific idioms that don't map cleanly via
  simple API renaming (e.g., in-place operations, lifecycle management,
  functional state patterns).

  Args:
      rewriter: The parent Rewriter instance containing context and config.
      original: The original CST node (before traversal).
      updated: The updated CST node (after children handling).
      func_name: The resolved fully qualified name of the function, or None if unresolved.

  Returns:
      Tuple[bool, cst.CSTNode]: A tuple containing:

      -   ``handled (bool)``: If True, the transformation is complete and should return early.
      -   ``node (cst.CSTNode)``: The result node (or updated node if not handled).
  """

  # 1. Functional 'apply' unwrapping (Dynamic Trait)
  # Checks if source uses functional patterns (e.g. Flax 'apply') that need unwrapping
  source_traits = rewriter._get_source_traits()
  unwrap_method = source_traits.functional_execution_method

  if is_functional_apply(original, unwrap_method):
    if isinstance(updated.func, cst.Attribute):
      receiver = updated.func.value
      # Strip the first argument (variables/params) and use receiver as callable
      new_args = updated.args[1:] if len(updated.args) > 0 else []
      result_node = updated.with_changes(func=receiver, args=new_args)
      log_diff("Functional Unwrap", original, result_node)
      return True, result_node

  # 2. Plugin Check (Explicit Requirement or ODL In-Place Metadata)
  # If the known function name maps to a plugin-required stub, we flag it.
  plugin_claim = False
  is_inplace = False

  if func_name:
    mapping = rewriter._get_mapping(func_name, silent=True)
    if mapping and "requires_plugin" in mapping:
      plugin_claim = True

    # Check ODL metadata for inplace flag
    defn = rewriter.semantics.get_definition(func_name)
    if defn:
      _, details = defn
      if details.get("is_inplace"):
        is_inplace = True

  # 2b. Heuristic Plugin: In-Place Unrolling
  # Triggers if:
  # A) 'is_inplace' is set in ODL metadata
  # B) Method ends with '_' (heuristic) AND isn't claimed by another plugin
  should_unroll = False
  if is_inplace:
    should_unroll = True
  elif not plugin_claim and func_name and func_name.endswith("_") and not func_name.startswith("__"):
    should_unroll = True

  if should_unroll:
    hook = get_hook("unroll_inplace_ops")
    if hook:
      new_node = hook(updated, rewriter.ctx)
      if new_node != updated:
        log_diff("In-place Unroll", updated, new_node)
        # If transformed, we return it as final result for this phase
        return True, new_node

  # 3. Lifecycle Method Handling (Strip/Warn)
  strip_set, warn_set = rewriter._get_source_lifecycle_lists()

  if not plugin_claim and isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
    method_name = original.func.attr.value

    if method_name in strip_set:
      if isinstance(updated.func, cst.Attribute):
        # Identity transform: x.cuda() -> x
        rewriter._report_warning(f"Stripped framework-specific lifecycle method '.{method_name}()'.")
        result_node = updated.func.value
        log_diff("Lifecycle Strip", original, result_node)
        return True, result_node

    if method_name in warn_set:
      if isinstance(updated.func, cst.Attribute):
        # Identity transform with warning
        rewriter._report_warning(f"Ignored model state method '.{method_name}()'.")
        result_node = updated.func.value
        log_diff("Lifecycle Warn", original, result_node)
        return True, result_node

  # 4. Stateful Call (e.g. self.layer(x) -> self.layer.apply(params, x))
  if func_name and rewriter._is_stateful(func_name):
    fw_config = rewriter.semantics.get_framework_config(rewriter.target_fw)
    stateful_spec = fw_config.get("stateful_call")
    if stateful_spec:
      result_node = rewrite_stateful_call(rewriter, updated, func_name, stateful_spec)
      log_diff("State Mechanism", original, result_node)
      return True, result_node

  return False, updated


def resolve_implicit_method(rewriter: Any, original: cst.Call, func_name: Optional[str]) -> Optional[str]:
  """
  Attempts to resolve method calls on objects to full API paths.

  Example:
      ``x.view()`` -> ``torch.Tensor.view``

  Logic:

  1.  **Type Inference**: Consults the Symbol Table (if available) to determine
      the type of the receiver object (e.g., it is a Tensor).
  2.  **Heuristic Fallback**: Checks implicit root classes (e.g. ``torch.Tensor``)
      defined in the source traits to guess if the method belongs to them.

  Args:
      rewriter: The calling rewriter instance.
      original: The original CST node structure.
      func_name: The currently resolved name (usually None or the failed name).

  Returns:
      Optional[str]: Resolved fully qualified name, or None if no match found.
  """
  # Check if it looks like a method call: object.method(...)
  if isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
    receiver = original.func.value
    leaf_method = original.func.attr.value

    # Ensure we aren't misinterpreting module alias `torch.abs` as `obj.abs`
    is_module = rewriter._is_module_alias(receiver)
    # Check for 'self'
    is_self = isinstance(receiver, cst.Name) and receiver.value == "self"

    if not is_self and not is_module:
      # --- 1. Symbol Table Inference ---
      if hasattr(rewriter, "symbol_table") and rewriter.symbol_table:
        sym_type = rewriter.symbol_table.get_type(receiver)
        if sym_type:
          # Construct API path based on inferred type
          # e.g., if type is TensorType(framework='torch'), try 'torch.Tensor.view'
          candidate_api = f"{sym_type.name}.{leaf_method}"

          # Handle Tensor special casing map if frameworks define 'Tensor' roots differently
          if "Tensor" in sym_type.name and hasattr(sym_type, "framework"):
            # 'torch.Tensor.view'
            candidate_api = f"{sym_type.framework}.Tensor.{leaf_method}"

          mapping = rewriter._get_mapping(candidate_api, silent=True)
          if mapping:
            return candidate_api

      # --- 2. Legacy Heuristic Fallback ---
      source_traits = rewriter._get_source_traits()
      implicit_roots = source_traits.implicit_method_roots

      for root in implicit_roots:
        candidate_api = f"{root}.{leaf_method}"
        # Silent lookup
        candidate_mapping = rewriter._get_mapping(candidate_api, silent=True)
        if candidate_mapping:
          # Found a match via implicit root
          return candidate_api

  return None
