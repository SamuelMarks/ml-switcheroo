"""
Function Invocation Rewriter.

This module provides the :class:`InvocationMixin`, a component of the
:class:`PivotRewriter` responsible for transforming function call nodes within
the Abstract Syntax Tree (AST).

It handles:
1.  **Functional Unwrapping**: Converting legacy `apply` patterns to object calls.
2.  **Plugin Dispatch**: Delegating complex logic to registered hook functions.
3.  **Lifecycle Management**: Stripping or warning about framework-specific methods.
4.  **Stateful Calls**: Rewriting object calls that manage internal state.
5.  **API Mapping**: Renaming functions based on semantic definitions.
6.  **Argument Normalization**: Reordering and renaming arguments.
7.  **Syntax Transformation**: Converting calls to infix operators, lambdas, or macros.
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
  Mixin for transforming :class:`libcst.Call` nodes.

  This class centralizes the logic for rewriting function invocations (`func(...)`).
  It integrates normalization, trait lookup, and various transformation strategies
  defined in the Semantic Knowledge Base.
  """

  def leave_Call(
    self, original: cst.Call, updated: cst.Call
  ) -> Union[cst.Call, cst.BinaryOperation, cst.UnaryOperation, cst.CSTNode]:
    """
    Visits and rewrites a function call node.

    The transformation pipeline proceeds as follows:

    1.  **Functional Unwrapping**: Checks if the call matches a functional execution
        pattern (e.g., ``layer.apply(...)``) and unwraps it if required by the
        source framework traits.
    2.  **Plugin & Heuristics**: Checks if specific plugins (e.g., in-place unrolling)
        need to intervene before standard semantic lookup.
    3.  **Lifecycle Handling**: Strips or warns about framework-specific lifecycle
        methods (e.g., ``.cuda()``, ``.eval()``) based on source traits.
    4.  **Stateful Object Calls**: Rewrites calls to stateful objects if required
        by the target framework configuration.
    5.  **Semantic Lookup**: Resolves the function name to an Abstract Operation ID.
        If no mapping is found, attempts implicit method resolution or returns early.
    6.  **Dispatch Rules**: Evaluates conditional logic to dynamically switch the
        target API based on argument values.
    7.  **Transformation Strategy**: Applies specific rewriting logic:
        *   **Infix**: Converts to operator syntax (e.g., ``add(a, b)`` -> ``a + b``).
        *   **Inline Lambda**: Wraps arguments in a lambda expression.
        *   **Plugin**: Delegates to a named plugin hook.
        *   **Macro**: Expands a template string.
        *   **Standard**: Renames the function and normalizes arguments.
    8.  **Output Processing**: Applies output adapters (indexing, casting) if defined.
    9.  **Init Logic**: Handles state threading for constructor calls if active.

    Args:
        original: The original CST node before child transformations.
        updated: The CST node after children have been visited/transformed.

    Returns:
        The transformed CST node (Call, BinaryOperation, or other expression).
    """
    result_node = updated
    func_name = self._get_qualified_name(original.func)

    # 0a. Functional 'apply' unwrapping (Dynamic Trait)
    source_traits = self._get_source_traits()
    unwrap_method = source_traits.functional_execution_method

    if is_functional_apply(original, unwrap_method):
      if isinstance(updated.func, cst.Attribute):
        receiver = updated.func.value
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
          if not isinstance(new_node, cst.Call):
            return new_node
          updated = new_node
          func_name = self._get_qualified_name(
            updated.operator if isinstance(updated, cst.BinaryOperation) else updated.func
          )

    # 0c. Lifecycle Method Handling
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

    # --- FALLBACK LOGIC: Implicit Methods ---
    if not mapping and isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
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
            definition = self.semantics.get_definition(candidate_api)
            if definition:
              self.ctx.current_op_id = definition[0]
            break

    # 3. Final Check
    if not mapping:
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

    self._handle_variant_imports(mapping)

    # Dispatch Rules
    if "dispatch_rules" in mapping and mapping["dispatch_rules"]:
      dispatched_api = evaluate_dispatch_rules(self, original, mapping["dispatch_rules"], details)
      if dispatched_api:
        mapping = mapping.copy()
        mapping["api"] = dispatched_api

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
      target_api = mapping.get("api")

      # FIX: Handle missing API (unsupported ops) gracefully
      if not target_api:
        msg = mapping.get("missing_message", f"No mapping available for '{func_name}' -> '{self.target_fw}'")
        self._report_failure(msg)
        return updated

      try:
        norm_args = self._normalize_arguments(original, updated, details, mapping)
        new_func = self._create_name_node(target_api)
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
