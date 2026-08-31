"""Pre-processing Phase for Call Rewriting.

Handles functional unwrapping, plugin claims, and lifecycle method stripping.
Updated to remove dependencies on deleted legacy modules.
"""

from typing import Tuple, Optional, TYPE_CHECKING, Set

import libcst as cst

from ml_switcheroo.core.rewriter.calls.utils import (
  is_functional_apply,
  log_diff,
  rewrite_stateful_call,
)
from ml_switcheroo.core.hooks import get_hook

if TYPE_CHECKING:
  # Structural typing for rewriter to avoid circular import
  class SourceTraitsDummy:
    """Structural type representation of a hook context."""

    functional_execution_method: str
    implicit_method_roots: list

  class SemanticManagerDummy:
    """Structural type representation of a hook context."""

    def get_definition(self, func_name: str) -> Optional[Tuple[str, dict]]:
      """Structural method signature."""
      ...

    def get_framework_config(self, target_fw: str) -> dict:
      """Structural method signature."""
      ...

  class HookContextDummy:
    """Structural type representation of a hook context."""

    pass

  class SignatureContextDummy:
    """Structural type representation of a hook context."""

    existing_args: set
    injected_args: list

  class RewriterContextDummy:
    """Structural type representation of a hook context."""

    hook_context: HookContextDummy
    symbol_table: "SymbolTableDummy"
    signature_stack: list[SignatureContextDummy]

  class SymbolTypeDummy:
    """Structural type representation of a hook context."""

    name: str
    framework: str

  class SymbolTableDummy:
    """Structural type representation of a hook context."""

    def get_type(self, node: cst.CSTNode) -> Optional[SymbolTypeDummy]:
      """Structural method signature."""
      ...

  class RewriterDummy:
    """Structural type representation of a hook context."""

    source_traits: SourceTraitsDummy
    semantics: SemanticManagerDummy
    target_fw: str
    source_fw: str
    context: RewriterContextDummy

    def _get_source_traits(self) -> SourceTraitsDummy:
      """Structural method signature."""
      ...

    def _get_mapping(self, func_name: str, silent: bool = False) -> Optional[dict]:
      """Structural method signature."""
      ...

    def _get_source_lifecycle_lists(self) -> Tuple[Set[str], Set[str]]:
      """Structural method signature."""
      ...

    def _report_warning(self, msg: str) -> None:
      """Structural method signature."""
      ...

    def _is_stateful(self, func_name: str) -> bool:
      """Structural method signature."""
      ...

    def _is_module_alias(self, name: cst.BaseExpression) -> bool:
      """Structural method signature."""
      ...

    def _get_target_traits(self) -> "SourceTraitsDummy":
      """Structural method signature."""
      ...

    def _create_dotted_name(self, name: str) -> cst.Attribute:
      """Structural method signature."""
      ...


def handle_pre_checks(
  rewriter: "RewriterDummy", original: cst.Call, updated: cst.Call, func_name: Optional[str]
) -> Tuple[bool, cst.BaseExpression]:
  """Execute pre-lookup checks and transformations.

  Args:
      rewriter: The calling transformer (duck typing: needs _get_source_traits,
                _report_warning, _is_stateful, semantics, target_fw, ctx).
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
      source_traits = rewriter._get_source_traits()

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

        result_node = updated.with_changes(func=receiver, args=new_args)
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
      mapping = rewriter._get_mapping(func_name, silent=True)

    if mapping and "requires_plugin" in mapping:
      plugin_claim = True

    # Check ODL metadata for inplace flag
    defn = rewriter.semantics.get_definition(func_name)
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
      new_node = hook(updated, rewriter.context.hook_context)
      if new_node != updated:
        log_diff("In-place Unroll", updated, new_node)
        return True, new_node

  # 3. Lifecycle Method Handling (Strip/Warn)
  if hasattr(rewriter, "_get_source_lifecycle_lists"):
    strip_set, warn_set = rewriter._get_source_lifecycle_lists()

    if not plugin_claim and isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
      method_name = original.func.attr.value

      if method_name in strip_set:
        if isinstance(updated.func, cst.Attribute):
          rewriter._report_warning(f"Stripped framework-specific lifecycle method '.{method_name}()'.")
          result_node = updated.func.value
          log_diff("Lifecycle Strip", original, result_node)
          return True, result_node

      if method_name in warn_set:
        if isinstance(updated.func, cst.Attribute):
          rewriter._report_warning(f"Ignored model state method '.{method_name}()'.")
          result_node = updated.func.value
          log_diff("Lifecycle Warn", original, result_node)
          return True, result_node

  # 4. Stateful Call
  if func_name and hasattr(rewriter, "_is_stateful") and rewriter._is_stateful(func_name):
    fw_config = rewriter.semantics.get_framework_config(rewriter.target_fw)
    stateful_spec = fw_config.get("stateful_call")
    if stateful_spec:
      result_node = rewrite_stateful_call(rewriter, updated, func_name, stateful_spec)  # type: ignore[arg-type]
      log_diff("State Mechanism", original, result_node)
      return True, result_node

  return False, updated


def resolve_implicit_method(rewriter: "RewriterDummy", original: cst.Call, func_name: Optional[str]) -> Optional[str]:
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
      is_module = rewriter._is_module_alias(receiver)

    if not is_self and not is_module:
      # --- 1. Symbol Table Inference ---
      if hasattr(rewriter, "context") and rewriter.context.symbol_table:
        sym_type = rewriter.context.symbol_table.get_type(receiver)
        if sym_type:
          # Construct API path based on inferred type
          candidate_api = f"{sym_type.name}.{leaf_method}"

          if "Tensor" in sym_type.name and hasattr(sym_type, "framework"):
            candidate_api = f"{sym_type.framework}.Tensor.{leaf_method}"

          if hasattr(rewriter, "_get_mapping"):
            mapping = rewriter._get_mapping(candidate_api, silent=True)
            if mapping:
              return candidate_api

      # --- 2. Legacy Heuristic Fallback ---
      if hasattr(rewriter, "_get_target_traits"):
        # Note: Implicit roots usually belong to SOURCE traits
        if hasattr(rewriter, "source_traits"):
          traits = rewriter.source_traits  # type: ignore[assignment]
        else:
          # Fallback if property missing (shouldn't happen in ApiPass)
          config_dict = rewriter.semantics.get_framework_config(rewriter.source_fw)
          from ml_switcheroo.semantics.schema import StructuralTraits

          traits = StructuralTraits.model_validate(config_dict.get("traits", {}))  # type: ignore[assignment]

        implicit_roots = traits.implicit_method_roots

        for root in implicit_roots:
          candidate_api = f"{root}.{leaf_method}"
          if hasattr(rewriter, "_get_mapping"):
            mapping = rewriter._get_mapping(candidate_api, silent=True)
            if mapping:
              return candidate_api

  return None
