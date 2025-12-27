"""
CallMixin Definition.

The primary visitor logic for handling `Call` and `Assign` nodes in the AST.
This module orchestrates the transformation pipeline by delegating specific tasks
(dispatch, reshaping, lifecycle management) to helper modules.

Updates:
- Implemented `auto_strip_magic_args` logic in `leave_Call` to dynamically
  remove state arguments (rngs, key) based on global knowledge base aggregation.
- Implemented dynamic detection of functional execution patterns via `StructuralTraits.functional_execution_method`.
"""

from typing import Union, Set, Tuple, List
import libcst as cst

from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.core.rewriter.normalization import NormalizationMixin
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.utils.node_diff import capture_node_source

# Import split logic
from ml_switcheroo.core.rewriter.calls.dispatch import evaluate_dispatch_rules
from ml_switcheroo.core.rewriter.calls.transformers import (
  rewrite_as_infix,
  rewrite_as_inline_lambda,
  rewrite_as_macro,
  apply_output_adapter,
  apply_index_select,
)
from ml_switcheroo.core.rewriter.calls.utils import (
  is_functional_apply,
  rewrite_stateful_call,
  inject_kwarg,
  strip_kwarg,
  is_super_call,
  is_builtin,
  log_diff,
  compute_permutation,
  inject_permute_call,
)


class CallMixin(NormalizationMixin, BaseRewriter):
  """
  Mixin for transforming Call nodes and unpacking Assignments.

  Responsible for:
  1.  Handling functional `apply` patterns (Flax/Custom).
  2.  Lifecycle method stripping (`.to()`, `.cuda()`).
  3.  Plugin dispatch.
  4.  Standard API pivoting (Lookup -> Normalize -> Rewrite).
  5.  Output Transformation (Indexing, Casting).
  """

  # Internal cache for traits to avoid lookup overhead per call
  _cached_source_traits: StructuralTraits = None
  _cached_target_traits: StructuralTraits = None

  def _get_source_traits(self) -> StructuralTraits:
    """
    Lazily loads and caches the StructuralTraits of the SOURCE framework.
    """
    if self._cached_source_traits:
      return self._cached_source_traits

    config_dict = self.semantics.get_framework_config(self.source_fw)
    if config_dict and "traits" in config_dict:
      self._cached_source_traits = StructuralTraits.model_validate(config_dict["traits"])
    else:
      self._cached_source_traits = StructuralTraits()
    return self._cached_source_traits

  def _get_source_lifecycle_lists(self) -> Tuple[Set[str], Set[str]]:
    """
    Lazily loads the lifecycle strip/warn lists from the SOURCE framework config.
    """
    traits = self._get_source_traits()
    return (
      set(traits.lifecycle_strip_methods),
      set(traits.lifecycle_warn_methods),
    )

  def _get_target_traits(self) -> StructuralTraits:
    """Lazily loads properties of the TARGET framework."""
    if self._cached_target_traits:
      return self._cached_target_traits

    config_dict = self.semantics.get_framework_config(self.target_fw)
    if config_dict and "traits" in config_dict:
      self._cached_target_traits = StructuralTraits.model_validate(config_dict["traits"])
    else:
      self._cached_target_traits = StructuralTraits()

    return self._cached_target_traits

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """
    Handles assignment unwrapping for Functional -> OOP transitions.

    Scenario: `y, updates = layer.apply(vars, x)`
    Target:   `y = layer(x)` (NNX/Torch style)

    Reads `functional_execution_method` from Source Traits to detect pattern.
    """
    if not isinstance(original_node.value, cst.Call):
      return super().leave_Assign(original_node, updated_node)

    # Dynamic detection based on source trait (e.g. "apply", "call_fn")
    source_traits = self._get_source_traits()
    unwrap_method = source_traits.functional_execution_method

    if is_functional_apply(original_node.value, unwrap_method):
      if len(updated_node.targets) == 1:
        target = updated_node.targets[0].target
        if isinstance(target, (cst.Tuple, cst.List)):
          elements = target.elements
          if len(elements) > 0:
            primary_target = elements[0].value
            new_target = cst.AssignTarget(target=primary_target)

            new_node = updated_node.with_changes(targets=[new_target])
            get_tracer().log_mutation(
              "Assignment Unwrapping",
              capture_node_source(original_node),
              capture_node_source(new_node),
            )
            return new_node

    return super().leave_Assign(original_node, updated_node)

  def leave_Call(
    self, original: cst.Call, updated: cst.Call
  ) -> Union[cst.Call, cst.BinaryOperation, cst.UnaryOperation, cst.CSTNode]:
    """
    Rewrites function calls with detailed Trace Logging.
    Implements Logic 4: Layer Init State Threading.
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

    if not func_name:
      # Fix: Prevent strict mode failure on super() calls
      if is_super_call(original):
        return updated

      # Try to recover: if this is a method call on an unknown object, try resolving just the method name
      # This enables x.float() -> CastFloat without knowing x is a Tensor
      if isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
        func_name = f"__method_fallback__.{original.func.attr.value}"  # Placeholder
      else:
        if self.strict_mode:
          self._report_failure("Could not resolve function name")
        return updated

    # 2. Standard API Rewrite Setup
    mapping = self._get_mapping(func_name)

    # FIX: Fallback lookup for methods on unknown objects (e.g. x.float() where x is unknown type)
    if not mapping and isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
      receiver = original.func.value
      is_self = isinstance(receiver, cst.Name) and receiver.value == "self"

      if not is_self and not self._is_module_alias(receiver):
        leaf_method = original.func.attr.value
        # Try fallback lookup using the guessed API. Use silent=True to avoid reporting failure logic twice.
        full_api_guess = f"torch.Tensor.{leaf_method}"
        mapping = self._get_mapping(full_api_guess, silent=True)
        if mapping:
          func_name = full_api_guess

    if not mapping:
      # --- TRACE: Detailed Decision Log ---
      if not is_builtin(func_name) and not func_name.startswith("__method_fallback__"):
        get_tracer().log_inspection(
          node_str=func_name,
          outcome="Skipped",
          detail="No Entry in Semantics Knowledge Base",
        )
      # Re-check strict failure reporting for the ORIGINAL name if we skipped fallback reporting
      if self.strict_mode and func_name and func_name.startswith(f"{self.source_fw}."):
        self._report_failure(f"API '{func_name}' not found in semantics.")

      return updated

    lookup = self.semantics.get_definition(func_name)

    if not lookup:
      # Should not happen if mapping is present, unless broken state
      return updated

    abstract_id, details = lookup
    self.ctx.current_op_id = abstract_id  # Set context for plugins

    # --- Feature 13: Dynamic Import Injection ---
    self._handle_variant_imports(mapping)

    # --- FEATURE: Conditional API Dispatch ---
    if "dispatch_rules" in mapping and mapping["dispatch_rules"]:
      dispatched_api = evaluate_dispatch_rules(self, original, mapping["dispatch_rules"], details)
      if dispatched_api:
        mapping = mapping.copy()
        mapping["api"] = dispatched_api

    # 2a. Infix / Prefix Transformation
    if mapping.get("transformation_type") == "infix":
      try:
        norm_args = self._normalize_arguments(original, updated, details, mapping)
        result_node = rewrite_as_infix(
          original,
          norm_args,
          mapping.get("operator"),
          details.get("std_args", []),
        )
      except (ValueError, IndexError) as e:
        self._report_failure(f"Infix/Prefix transformation failed: {e}")
        return updated

    # 2b. Inline Lambda Transformation
    elif mapping.get("transformation_type") == "inline_lambda":
      try:
        norm_args = self._normalize_arguments(original, updated, details, mapping)
        result_node = rewrite_as_inline_lambda(mapping["api"], norm_args)
      except Exception as e:
        self._report_failure(f"Inline lambda transformation failed: {e}")
        return updated

    # 2c. Plugin Dispatch
    elif "requires_plugin" in mapping:
      plugin_name = mapping["requires_plugin"]
      hook = get_hook(plugin_name)
      if hook:
        result_node = hook(updated, self.ctx)
      else:
        self._report_failure(f"Missing required plugin: '{plugin_name}'")
        return updated

    # 2d. Macro Template Expansion
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

    # 2e. Standard Function Rewrite
    else:
      try:
        norm_args = self._normalize_arguments(original, updated, details, mapping)
        new_func = self._create_name_node(mapping["api"])
        result_node = updated.with_changes(func=new_func, args=norm_args)

        # --- Feature: Tensor Layout Permutation Input Injection ---
        # Check if the variant defines a layout map
        if "layout_map" in mapping and mapping["layout_map"]:
          layout_map = mapping["layout_map"]
          std_args_raw = details.get("std_args", [])

          idx = 0
          modified_args = list(result_node.args)

          for item in std_args_raw:
            arg_name = None
            if isinstance(item, (list, tuple)):
              arg_name = item[0]
            elif isinstance(item, dict):
              arg_name = item.get("name")
            elif isinstance(item, str):
              arg_name = item

            if arg_name and arg_name in layout_map:
              rule = layout_map[arg_name]
              if "->" in rule:
                src_l, tgt_l = rule.split("->")
                perm_indices = compute_permutation(src_l.strip(), tgt_l.strip())

                if perm_indices and idx < len(modified_args):
                  # Wrap input
                  original_arg = modified_args[idx]
                  wrapped_val = inject_permute_call(
                    original_arg.value,
                    perm_indices,
                    self.semantics,
                    self.target_fw,
                  )
                  modified_args[idx] = original_arg.with_changes(value=wrapped_val)

            idx += 1

          result_node = result_node.with_changes(args=modified_args)

          # --- Handle Return Permutation ---
          if "return" in layout_map:
            rule = layout_map["return"]
            if "->" in rule:
              src_l, tgt_l = rule.split("->")
              perm_indices = compute_permutation(src_l.strip(), tgt_l.strip())
              if perm_indices:
                result_node = inject_permute_call(
                  result_node,
                  perm_indices,
                  self.semantics,
                  self.target_fw,
                )

      except ValueError:
        self._report_failure("Argument normalization failed")
        return updated

    # 3. Output Normalization (Post-Processing)
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

    # 3b. Output Casting logic (Feature 12)
    if "output_cast" in mapping and mapping["output_cast"]:
      try:
        type_node = self._create_dotted_name(mapping["output_cast"])
        # Wrap result_node.astype(type_node)
        result_node = cst.Call(
          func=cst.Attribute(value=result_node, attr=cst.Name("astype")),
          args=[cst.Arg(value=type_node)],
        )
      except Exception as e:
        self._report_failure(f"Output casting failed: {e}")
        return updated

    # 4. Class Construction Logic (Generic State Threading)
    # Check if we are inside an __init__ method of what looks like a Framework Class (Module)
    if self._signature_stack and self._signature_stack[-1].is_init and self._signature_stack[-1].is_module_method:
      origins = getattr(self.semantics, "_key_origins", {})
      tier = origins.get(abstract_id)
      traits = self._get_target_traits()

      is_neural_candidate = tier == SemanticTier.NEURAL.value
      force_detect = False

      # Check if any argument in the call matches a 'strip' target,
      # which implies this call creates a sub-layer that needs handling
      if isinstance(result_node, cst.Call):
        # Build set of magic args to check against
        magic_args = set(traits.strip_magic_args)
        if traits.auto_strip_magic_args and hasattr(self.semantics, "known_magic_args"):
          magic_args.update(self.semantics.known_magic_args)

        for arg in result_node.args:
          if arg.keyword and arg.keyword.value in magic_args:
            force_detect = True
            break

      if is_neural_candidate or force_detect:
        if isinstance(result_node, cst.Call):
          # Inject Generic Magic Args (e.g. rngs=rngs, ctx=ctx)
          # We iterate over the configured list of (arg_name, type)
          for arg_name, _ in traits.inject_magic_args:
            # Assume the variable name in scope matches the keyword -> arg=arg
            result_node = inject_kwarg(result_node, arg_name, arg_name)

          # Strip Generic Magic Args (e.g. rngs, key)
          # Implement Decoupled Auto-Strip Logic logic
          args_to_strip = set(traits.strip_magic_args)
          if traits.auto_strip_magic_args and hasattr(self.semantics, "known_magic_args"):
            # Aggregated list from all frameworks
            args_to_strip.update(self.semantics.known_magic_args)

            # IMPORTANT: Do not strip args that are native/required for THIS framework
            native_args = {a[0] for a in traits.inject_magic_args}
            args_to_strip = args_to_strip - native_args

          for arg_name in args_to_strip:
            result_node = strip_kwarg(result_node, arg_name)

    log_diff(f"Operation ({abstract_id})", original, result_node)

    return result_node
