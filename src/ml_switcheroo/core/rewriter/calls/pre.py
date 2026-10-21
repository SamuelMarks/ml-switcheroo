"""Pre-processing Phase for Call Rewriting.

Handles functional unwrapping, plugin claims, and lifecycle method stripping.
Updated to remove dependencies on deleted legacy modules.
"""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply, log_diff, rewrite_stateful_call

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.interface import RewriterHost


class PreRewriterProtocol(typing.Protocol):
  """Minimal rewriter protocol required by handle_pre_checks and resolve_implicit_method."""

  @property
  def source_framework(self) -> str:
    """Return the source framework identifier.

    Returns:
        Source framework identifier string.
    """
    ...


def handle_pre_checks(
  rewriter: "RewriterHost | PreRewriterProtocol",
  original: cst.Call,
  updated: cst.Call,
  func_name: str | None,
) -> tuple[bool, cst.BaseExpression]:
  """Execute pre-lookup checks and transformations.

  Args:
      rewriter: The calling transformer (duck typing: needs _get_source_traits,
                _report_warning, _is_stateful, semantics, target_framework, ctx).
      original: The original CST node.
      updated: The updated CST node.
      func_name: Resolved function name.

  Returns:
      Tuple(handled, result_node).

  """
  # 1. Functional 'apply' unwrapping (Dynamic Trait)
  # FIX: Check for property 'source_traits' OR method '_get_source_traits' to be robust
  has_traits = hasattr(rewriter, "source_traits") or hasattr(rewriter, "_get_source_traits")

  if has_traits:
    if hasattr(rewriter, "source_traits"):
      source_traits = rewriter.source_traits
    else:
      source_traits = getattr(rewriter, "_get_source_traits")()

    unwrap_method = source_traits.functional_execution_method

    if is_functional_apply(original, unwrap_method):
      if isinstance(updated.func, cst.Attribute):
        receiver = updated.func.value
        # Strip the first argument (variables/params) and use receiver as callable
        # e.g. layer.apply(vars, x) -> layer(x)
        # Check args exist
        if len(updated.args) > 0:
          new_args = updated.args[1:]
        else:
          new_args = []

        result_node: cst.BaseExpression = updated.with_changes(func=receiver, args=new_args)
        log_diff("Functional Unwrap", original, result_node)
        return True, result_node
  else:
    # Fallback if traits not available
    _ = True

  # 2. Plugin Check (Explicit Requirement or ODL In-Place Metadata)
  plugin_claim = False
  is_inplace = False

  if func_name:
    # Use rewriter._get_mapping logic if available
    mapping = None
    if hasattr(rewriter, "_get_mapping"):
      mapping = getattr(rewriter, "_get_mapping")(func_name, silent=True)

    if mapping and "requires_plugin" in mapping:
      plugin_claim = True

    # Check ODL metadata for inplace flag
    sem_obj = getattr(rewriter, "semantics", None)
    defn = sem_obj.get_definition(func_name) if sem_obj is not None else None
    if defn:
      _, details = defn
      if details.get("is_inplace"):
        is_inplace = True

  # 2b. Heuristic: In-Place Unrolling
  should_unroll = False
  if is_inplace:
    should_unroll = True
  elif not plugin_claim and func_name and func_name.endswith("_") and not func_name.startswith("__"):
    should_unroll = True

  if should_unroll:
    hook = get_hook("unroll_inplace_ops")
    if hook:
      ctx_obj = getattr(rewriter, "context", None)
      new_node = hook(updated, getattr(ctx_obj, "hook_context"))
      if isinstance(new_node, cst.BaseExpression) and new_node != updated:
        log_diff("In-place Unroll", updated, new_node)
        return True, new_node

  # 3. Lifecycle Method Handling (Strip/Warn)
  if hasattr(rewriter, "_get_source_lifecycle_lists"):
    strip_set, warn_set = getattr(rewriter, "_get_source_lifecycle_lists")()

    if not plugin_claim and isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
      method_name = original.func.attr.value

      if method_name in strip_set:
        if isinstance(updated.func, cst.Attribute):
          getattr(rewriter, "_report_warning")(f"Stripped framework-specific lifecycle method '.{method_name}()'.")
          result_node = updated.func.value
          log_diff("Lifecycle Strip", original, result_node)
          return True, result_node

      if method_name in warn_set:
        if isinstance(updated.func, cst.Attribute):
          getattr(rewriter, "_report_warning")(f"Ignored model state method '.{method_name}()'.")
          result_node = updated.func.value
          log_diff("Lifecycle Warn", original, result_node)
          return True, result_node

  # 4. Stateful Call
  if func_name and hasattr(rewriter, "_is_stateful") and getattr(rewriter, "_is_stateful")(func_name):
    sem_obj = getattr(rewriter, "semantics", None)
    fw_config = sem_obj.get_framework_config(getattr(rewriter, "target_framework")) if sem_obj is not None else None
    raw_stateful = fw_config.get("stateful_call") if fw_config else None
    if isinstance(raw_stateful, dict) and raw_stateful:
      stateful_spec: dict[str, str] = {str(k): str(v) for k, v in raw_stateful.items()}
      result_node = rewrite_stateful_call(rewriter, updated, func_name, stateful_spec)
      log_diff("State Mechanism", original, result_node)
      return True, result_node

  return False, updated


def resolve_implicit_method(
  rewriter: "RewriterHost | PreRewriterProtocol",
  original: cst.Call,
  func_name: str | None,
) -> str | None:
  """Attempt to resolve method calls on objects to full API paths.

  Args:
      rewriter: The calling transformer object.
      original: The original Call node from libcst.
      func_name: The optional function name string.

  Returns:
      The resolved full API path if found, otherwise None.

  """
  if isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
    receiver = original.func.value
    leaf_method = original.func.attr.value

    # Check for 'self'
    is_self = isinstance(receiver, cst.Name) and receiver.value == "self"

    # Check if module alias (requires rewriter alias checker)
    is_module = False
    if hasattr(rewriter, "_is_module_alias"):
      is_module = getattr(rewriter, "_is_module_alias")(receiver)

    if not is_self and not is_module:
      # --- 1. Symbol Table Inference ---
      ctx_obj = getattr(rewriter, "context", None)
      sym_table = getattr(ctx_obj, "symbol_table") if ctx_obj is not None else None
      if sym_table:
        sym_type = sym_table.get_type(receiver)
        if sym_type:
          # Construct API path based on inferred type
          candidate_api = f"{sym_type.name}.{leaf_method}"

          if "Tensor" in sym_type.name and hasattr(sym_type, "framework"):
            candidate_api = f"{sym_type.framework}.Tensor.{leaf_method}"

          if hasattr(rewriter, "_get_mapping"):
            mapping = getattr(rewriter, "_get_mapping")(candidate_api, silent=True)
            if mapping:
              return candidate_api

      # --- 2. Legacy Heuristic Fallback ---
      if hasattr(rewriter, "_get_target_traits"):
        # Note: Implicit roots usually belong to SOURCE traits
        if hasattr(rewriter, "source_traits"):
          traits = getattr(rewriter, "source_traits")
        else:
          # Fallback if property missing (shouldn't happen in ApiPass)
          config_dict = getattr(rewriter, "semantics").get_framework_config(rewriter.source_framework)
          from ml_switcheroo.semantics.schema import StructuralTraits

          traits = StructuralTraits.model_validate(config_dict.get("traits", {}))

        implicit_roots = traits.implicit_method_roots

        for root in implicit_roots:
          candidate_api = f"{root}.{leaf_method}"
          if hasattr(rewriter, "_get_mapping"):
            mapping = getattr(rewriter, "_get_mapping")(candidate_api, silent=True)
            if mapping:
              return candidate_api

  return None
