"""
Function Invocation Rewriter.

This module provides the ``InvocationMixin``, the core component responsible for
rewriting `Call` nodes. It orchestrates a complex pipeline of transformations:

1.  **Functional Unwrapping**: Removing `apply` or `call_fn` wrappers.
2.  **Plugin Dispatch**: Delegating complex logic to registered hooks.
3.  **Lifecycle Management**: Stripping framework-specific methods like `.to()` or `.cuda()`.
4.  **State Management**: Rewriting calls to stateful objects/layers.
5.  **Standard API Pivoting**:
    -   Resolving abstract operations.
    -   Evaluating conditional dispatch rules.
    -   Normalizing arguments via the Semantic Specification.
    -   Applying Tensor Layout Permutations.
6.  **Output Transformation**: Applying casting or indexing adapters to the result.
7.  **State Threading**: Injecting state arguments (like `rngs`) in constructors.
"""

from typing import Union

import libcst as cst

from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.core.rewriter.calls.dispatch import evaluate_dispatch_rules
from ml_switcheroo.core.rewriter.calls.traits_cache import TraitsCachingMixin
from ml_switcheroo.core.rewriter.calls.transformers import (
  apply_index_select,
  apply_output_adapter,
  rewrite_as_infix,
  rewrite_as_inline_lambda,
  rewrite_as_macro,
)
from ml_switcheroo.core.rewriter.calls.utils import (
  compute_permutation,
  inject_kwarg,
  inject_permute_call,
  is_builtin,
  is_functional_apply,
  is_super_call,
  log_diff,
  rewrite_stateful_call,
  strip_kwarg,
)
from ml_switcheroo.core.rewriter.normalization import NormalizationMixin
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.enums import SemanticTier


class InvocationMixin(NormalizationMixin, TraitsCachingMixin):
  """
  Mixin for transforming Call nodes (`func(...)`).

  This class contains the primary logic for mapping function calls between frameworks.
  It relies on `NormalizationMixin` for argument mapping and `TraitsCachingMixin` for
  accessing framework configurations.
  """

  def leave_Call(
    self, original: cst.Call, updated: cst.Call
  ) -> Union[cst.Call, cst.BinaryOperation, cst.UnaryOperation, cst.CSTNode]:
    """
    Rewrites function calls with detailed Trace Logging.

    The transformation pipeline proceeds in stages:
    1.  **Structure**: Unwrap functional `apply` calls, handle plugins, handle lifecycle methods.
    2.  **API Resolution**: Lookup the function name in the Semantic Knowledge Base.
    3.  **Argument Rewrite**: Normalize arguments, check dispatch rules, applying pivots.
    4.  **Transformation**: Rewrites call as Infix, Inline Lambda, Macro, or Standard Call.
    5.  **Output**: Applies indexing, output adapters, or casting.
    6.  **Context**: Helper logic for specific contexts like `__init__` (state threading).

    Args:
        original: The original CST node (before children were visited).
        updated: The CST node with transformed children (arguments processed).

    Returns:
        The transformed CST node (Call, Expr, or other).
    """
    result_node = updated
    func_name = self._get_qualified_name(original.func)

    # 0a. Functional 'apply' unwrapping (Dynamic Trait)
    source_traits = self._get_source_traits()
    unwrap_method = source_traits.functional_execution_method

    if is_functional_apply(original, unwrap_method):
      if isinstance(updated.func, cst.Attribute):
        receiver = updated.func.value
        # Functional pattern: apply(variables, x) -> x, so we strip variables (arg 0)
        new_args = updated.args[1:] if len(updated.args) > 0 else []
        result_node = updated.with_changes(func=receiver, args=new_args)
        log_diff("Functional Unwrap", original, result_node)
        return result_node

    # 0b. Plugin Check
    plugin_claim = False
    if func_name:
      mapping = self._get_mapping(func_name)
      if mapping and "requires_plugin" in mapping:
        plugin_claim = True

    # 0b.2 Heuristic: In-Place Unrolling
    if not plugin_claim and func_name and func_name.endswith("_") and not func_name.startswith("__"):
      hook = get_hook("unroll_inplace_ops")
      if hook:
        new_node = hook(updated, self.ctx)
        if new_node != updated:
          log_diff("In-place Unroll (Heuristic)", updated, new_node)
          # If helper returned a BinaryOperation (e.g. x + y), we cannot process it as Call further
          if not isinstance(new_node, cst.Call):
            return new_node
          updated = new_node
          func_name = self._get_qualified_name(
            updated.operator if isinstance(updated, cst.BinaryOperation) else updated.func
          )

    # 0c. Lifecycle Method Handling (Data-Driven via Source Framework Traits)
    strip_set, warn_set = self._get_source_lifecycle_lists()

    if not plugin_claim and isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
      method_name = original.func.attr.value

      if method_name in strip_set:
        if isinstance(updated.func, cst.Attribute):
          self._report_warning(f"Stripped framework-specific lifecycle method '.{method_name}()'.")
          result_node = updated.func.value
          log_diff("Lifecycle Strip", original, result_node)
          return result_node

      if method_name in warn_set:
        if isinstance(updated.func, cst.Attribute):
          self._report_warning(f"Ignored model state method '.{method_name}()'.")
          result_node = updated.func.value
          log_diff("Lifecycle Warn", original, result_node)
          return result_node

    # 1. Stateful Object Usage
    if func_name and self._is_stateful(func_name):
      fw_config = self.semantics.get_framework_config(self.target_fw)
      stateful_spec = fw_config.get("stateful_call")
      if stateful_spec:
        result_node = rewrite_stateful_call(self, updated, func_name, stateful_spec)
        log_diff("State Mechanism", original, result_node)
        return result_node

    # 2. Standard API Rewrite Setup
    mapping = self._get_mapping(func_name) if func_name else None

    # --- FALLBACK LOGIC: Implicit Methods (Decoupled) ---
    # If mapping not found, check if this is a method call on a generic object
    # that matches an implicit root defined by the source framework traits.
    # e.g. x.float() -> check implicit method root "torch.Tensor" -> look up "torch.Tensor.float"
    if not mapping and isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
      # Ensure it's not a self-call or known module call
      receiver = original.func.value
      is_self = isinstance(receiver, cst.Name) and receiver.value == "self"

      if not is_self and not self._is_module_alias(receiver):
        leaf_method = original.func.attr.value
        implicit_roots = source_traits.implicit_method_roots

        for root in implicit_roots:
          candidate_api = f"{root}.{leaf_method}"
          # Use silent lookup to avoid logging failures for guesses
          candidate_mapping = self._get_mapping(candidate_api, silent=True)
          if candidate_mapping:
            mapping = candidate_mapping
            func_name = candidate_api
            # Update context op ID for plugins so they can read metadata
            definition = self.semantics.get_definition(candidate_api)
            if definition:
              self.ctx.current_op_id = definition[0]
            break

    # 3. Final Check
    if not mapping:
      # Fix super() strict error
      if is_super_call(original):
        return updated

      if func_name and not is_builtin(func_name):
        get_tracer().log_inspection(
          node_str=func_name,
          outcome="Skipped",
          detail="No Entry in Semantics Knowledge Base",
        )

      if self.strict_mode and func_name and func_name.startswith(f"{self.source_fw}."):
        self._report_failure(f"API '{func_name}' not found in semantics.")

      return updated

    # 4. Determine Definition Detail
    lookup = self.semantics.get_definition(func_name)
    if not lookup:
      return updated

    abstract_id, details = lookup
    self.ctx.current_op_id = abstract_id

    # Dynamic Imports
    self._handle_variant_imports(mapping)

    # Dispatch Rules
    if "dispatch_rules" in mapping and mapping["dispatch_rules"]:
      dispatched_api = evaluate_dispatch_rules(self, original, mapping["dispatch_rules"], details)
      if dispatched_api:
        mapping = mapping.copy()
        mapping["api"] = dispatched_api

    # Execute Transformation based on Type
    trans_type = mapping.get("transformation_type")

    if trans_type == "infix":
      try:
        norm_args = self._normalize_arguments(original, updated, details, mapping)
        result_node = rewrite_as_infix(original, norm_args, mapping.get("operator"), details.get("std_args", []))
      except (ValueError, IndexError) as e:
        self._report_failure(f"Infix/Prefix transformation failed: {e}")
        return updated

    elif trans_type == "inline_lambda":
      try:
        norm_args = self._normalize_arguments(original, updated, details, mapping)
        result_node = rewrite_as_inline_lambda(mapping["api"], norm_args)
      except Exception as e:
        self._report_failure(f"Inline lambda transformation failed: {e}")
        return updated

    elif "requires_plugin" in mapping:
      plugin_name = mapping["requires_plugin"]
      hook = get_hook(plugin_name)
      if hook:
        result_node = hook(updated, self.ctx)
      else:
        self._report_failure(f"Missing required plugin: '{plugin_name}'")
        return updated

    elif mapping.get("macro_template"):
      try:
        norm_args = self._normalize_arguments(original, updated, details, mapping)
        std_arg_names = []
        for item in details.get("std_args", []):
          if isinstance(item, (list, tuple)):
            std_arg_names.append(item[0])
          elif isinstance(item, dict):
            std_arg_names.append(item["name"])
          else:
            std_arg_names.append(item)
        result_node = rewrite_as_macro(mapping["macro_template"], norm_args, std_arg_names)
      except Exception as e:
        self._report_failure(f"Macro expansion failed: {e}")
        return updated

    else:
      # Standard Rewrite
      try:
        norm_args = self._normalize_arguments(original, updated, details, mapping)
        new_func = self._create_name_node(mapping["api"])
        result_node = updated.with_changes(func=new_func, args=norm_args)

        # Layout Permutation
        if "layout_map" in mapping and mapping["layout_map"]:
          layout_map = mapping["layout_map"]
          std_args_raw = details.get("std_args", [])
          idx = 0
          modified_args = list(result_node.args)

          for item in std_args_raw:
            arg_name = (
              item.get("name") if isinstance(item, dict) else (item[0] if isinstance(item, (list, tuple)) else item)
            )
            if arg_name and arg_name in layout_map:
              rule = layout_map[arg_name]
              if "->" in rule:
                src_l, tgt_l = rule.split("->")
                perm_indices = compute_permutation(src_l.strip(), tgt_l.strip())
                if perm_indices and idx < len(modified_args):
                  original_arg = modified_args[idx]
                  wrapped_val = inject_permute_call(original_arg.value, perm_indices, self.semantics, self.target_fw)
                  modified_args[idx] = original_arg.with_changes(value=wrapped_val)
            idx += 1

          result_node = result_node.with_changes(args=modified_args)

          # Return Permutation
          if "return" in layout_map:
            rule = layout_map["return"]
            if "->" in rule:
              src_l, tgt_l = rule.split("->")
              perm_indices = compute_permutation(src_l.strip(), tgt_l.strip())
              if perm_indices:
                result_node = inject_permute_call(result_node, perm_indices, self.semantics, self.target_fw)

      except ValueError:
        self._report_failure("Argument normalization failed")
        return updated

    # 5. Output Processing
    if "output_select_index" in mapping and mapping["output_select_index"] is not None:
      try:
        result_node = apply_index_select(result_node, mapping["output_select_index"])
      except Exception as e:
        self._report_failure(f"Output indexing failed: {e}")
        return updated
    elif "output_adapter" in mapping and mapping["output_adapter"]:
      try:
        result_node = apply_output_adapter(result_node, mapping["output_adapter"])
      except Exception as e:
        self._report_failure(f"Output adapter failed: {e}")
        return updated

    if "output_cast" in mapping and mapping["output_cast"]:
      try:
        type_node = self._create_dotted_name(mapping["output_cast"])
        result_node = cst.Call(
          func=cst.Attribute(value=result_node, attr=cst.Name("astype")), args=[cst.Arg(value=type_node)]
        )
      except Exception:
        pass

    # 6. Logic 4 (State Threading for Init)
    if self._signature_stack and self._signature_stack[-1].is_init and self._signature_stack[-1].is_module_method:
      origins = getattr(self.semantics, "_key_origins", {})
      tier = origins.get(abstract_id)
      traits = self._get_target_traits()
      is_neural = tier == SemanticTier.NEURAL.value
      force = False

      if isinstance(result_node, cst.Call):
        magic = set(traits.strip_magic_args)
        if traits.auto_strip_magic_args and hasattr(self.semantics, "known_magic_args"):
          magic.update(self.semantics.known_magic_args)

        for arg in result_node.args:
          if arg.keyword and arg.keyword.value in magic:
            force = True
            break

      if is_neural or force:
        if isinstance(result_node, cst.Call):
          for arg_name, _ in traits.inject_magic_args:
            result_node = inject_kwarg(result_node, arg_name, arg_name)

          args_to_strip = set(traits.strip_magic_args)
          if traits.auto_strip_magic_args and hasattr(self.semantics, "known_magic_args"):
            args_to_strip.update(self.semantics.known_magic_args)
            native = {a[0] for a in traits.inject_magic_args}
            args_to_strip -= native

          for arg_name in args_to_strip:
            result_node = strip_kwarg(result_node, arg_name)

    log_diff(f"Operation ({abstract_id})", original, result_node)
    return result_node
