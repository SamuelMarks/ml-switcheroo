"""Pre-processing Phase for Call Rewriting.

Handles functional unwrapping, plugin claims, and lifecycle method stripping
against the unified ``RewriterHost`` contract.
"""

import typing

import libcst as cst

from ml_switcheroo.analysis.symbol_types import TensorType
from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply, log_diff, rewrite_stateful_call

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.interface import RewriterHost


def handle_pre_checks(
  rewriter: "RewriterHost",
  original: cst.Call,
  updated: cst.Call,
  func_name: str | None,
) -> tuple[bool, cst.BaseExpression]:
  """Execute pre-lookup checks and transformations.

  Args:
      rewriter: Host CST rewriter conforming to ``RewriterHost``.
      original: The original CST ``Call`` node.
      updated: The updated CST ``Call`` node.
      func_name: Resolved function name if available.

  Returns:
      Tuple of ``(handled, result_node)``.
  """
  # 1. Functional 'apply' unwrapping (Dynamic Trait)
  unwrap_method = rewriter.source_traits.functional_execution_method
  if is_functional_apply(original, unwrap_method) and isinstance(updated.func, cst.Attribute):
    receiver = updated.func.value
    new_args = updated.args[1:] if len(updated.args) > 0 else []
    result_node: cst.BaseExpression = updated.with_changes(func=receiver, args=new_args)
    log_diff("Functional Unwrap", original, result_node)
    return True, result_node

  # 2. Plugin Check (Explicit Requirement or ODL In-Place Metadata)
  plugin_claim = False
  is_inplace = False

  if func_name:
    mapping = rewriter._get_mapping(func_name, silent=True)
    if isinstance(mapping, dict) and "requires_plugin" in mapping:
      plugin_claim = True

    defn = rewriter.semantics.get_definition(func_name)
    if defn is not None:
      _, details = defn
      if details.get("is_inplace"):
        is_inplace = True

  # 2b. Heuristic: In-Place Unrolling
  should_unroll = is_inplace or (
    not plugin_claim and func_name is not None and func_name.endswith("_") and not func_name.startswith("__")
  )
  if should_unroll:
    hook = get_hook("unroll_inplace_ops")
    if hook:
      new_node = hook(updated, rewriter.ctx)
      if isinstance(new_node, cst.BaseExpression) and new_node != updated:
        log_diff("In-place Unroll", updated, new_node)
        return True, new_node

  # 3. Lifecycle Method Handling (Strip/Warn)
  strip_set, warn_set = rewriter._get_source_lifecycle_lists()
  if not plugin_claim and isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
    method_name = original.func.attr.value

    if method_name in strip_set and isinstance(updated.func, cst.Attribute):
      rewriter._report_warning(f"Stripped framework-specific lifecycle method '.{method_name}()'.")
      result_node = updated.func.value
      log_diff("Lifecycle Strip", original, result_node)
      return True, result_node

    if method_name in warn_set and isinstance(updated.func, cst.Attribute):
      rewriter._report_warning(f"Ignored model state method '.{method_name}()'.")
      result_node = updated.func.value
      log_diff("Lifecycle Warn", original, result_node)
      return True, result_node

  # 4. Stateful Call
  if func_name and rewriter._is_stateful(func_name):
    from ml_switcheroo.semantics.resolution import query_framework_traits

    fw_traits = query_framework_traits(rewriter.semantics, rewriter.target_framework)
    if fw_traits.stateful_call:
      result_node = rewrite_stateful_call(rewriter, updated, func_name, dict(fw_traits.stateful_call))
      log_diff("State Mechanism", original, result_node)
      return True, result_node

  return False, updated


def resolve_implicit_method(
  rewriter: "RewriterHost",
  original: cst.Call,
  func_name: str | None,
) -> str | None:
  """Attempt to resolve method calls on objects to full API paths.

  Args:
      rewriter: Host CST rewriter conforming to ``RewriterHost``.
      original: The original ``Call`` node from LibCST.
      func_name: The optional unresolved function name string.

  Returns:
      The resolved full API path if found, otherwise ``None``.
  """
  if isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
    receiver = original.func.value
    leaf_method = original.func.attr.value

    is_self = isinstance(receiver, cst.Name) and receiver.value == "self"
    is_module = rewriter._is_module_alias(receiver)

    if not is_self and not is_module:
      # 1. Symbol Table Inference
      sym_table = rewriter.context.symbol_table
      if sym_table is not None:
        sym_type = sym_table.get_type(receiver)
        if sym_type is not None:
          sym_name = sym_type.name
          candidate_api = f"{sym_name}.{leaf_method}"

          if "Tensor" in sym_name and isinstance(sym_type, TensorType) and sym_type.framework:
            candidate_api = f"{sym_type.framework}.Tensor.{leaf_method}"

          if rewriter._get_mapping(candidate_api, silent=True):
            return candidate_api

      # 2. Implicit Method Roots Fallback
      for root in rewriter.source_traits.implicit_method_roots:
        candidate_api = f"{root}.{leaf_method}"
        if rewriter._get_mapping(candidate_api, silent=True):
          return candidate_api

  return None
