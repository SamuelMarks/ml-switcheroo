"""
Call Rewriting Logic.

Handles function invocation usage (`leave_Call`) and Assignment Unwrapping.
Includes functionality for:
- Lifecycle method stripping (Data-Driven via Source Framework Traits).
- Functional -> OOP Unwrapping.
- Stateful object management.
- Standard API pivots.
- Plugin dispatch.
- Output Normalization.
- **Logc 4: Layer Init State Threading** (Neural Tier detection & rngs injection).
- **Trace Instrumention** for visibility.
"""

from typing import Union, Set, Dict, Tuple
import libcst as cst

from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.core.rewriter.normalization import NormalizationMixin
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.utils.node_diff import capture_node_source, diff_nodes


class CallMixin(NormalizationMixin, BaseRewriter):
  """
  Mixin for transforming Call nodes and unpacking Assignments.
  """

  # Internal cache for traits to avoid lookup overhead per call
  _cached_source_traits: StructuralTraits = None
  _cached_target_traits: StructuralTraits = None

  def _get_source_lifecycle_lists(self) -> Tuple[Set[str], Set[str]]:
    """
    Lazily loads the lifecycle strip/warn lists from the SOURCE framework config.
    """
    if self._cached_source_traits:
      return (
        set(self._cached_source_traits.lifecycle_strip_methods),
        set(self._cached_source_traits.lifecycle_warn_methods),
      )

    # Look up config for the Source Framework (e.g. 'torch')
    config_dict = self.semantics.get_framework_config(self.source_fw)

    if config_dict and "traits" in config_dict:
      self._cached_source_traits = StructuralTraits.model_validate(config_dict["traits"])
    else:
      self._cached_source_traits = StructuralTraits()

    return (
      set(self._cached_source_traits.lifecycle_strip_methods),
      set(self._cached_source_traits.lifecycle_warn_methods),
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
    """
    if not isinstance(original_node.value, cst.Call):
      return super().leave_Assign(original_node, updated_node)

    if self._is_functional_apply(original_node.value):
      if len(updated_node.targets) == 1:
        target = updated_node.targets[0].target
        if isinstance(target, (cst.Tuple, cst.List)):
          elements = target.elements
          if len(elements) > 0:
            primary_target = elements[0].value
            new_target = cst.AssignTarget(target=primary_target)

            new_node = updated_node.with_changes(targets=[new_target])
            get_tracer().log_mutation(
              "Assignment Unwrapping", capture_node_source(original_node), capture_node_source(new_node)
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

    # 0a. Functional 'apply' unwrapping
    if self._is_functional_apply(original):
      if isinstance(updated.func, cst.Attribute):
        receiver = updated.func.value
        new_args = updated.args[1:] if len(updated.args) > 0 else []
        result_node = updated.with_changes(func=receiver, args=new_args)
        self._log_diff("Functional Unwrap", original, result_node)
        return result_node

    # 0b. Plugin Check
    plugin_claim = False
    if func_name:
      mapping = self._get_mapping(func_name)
      if mapping and "requires_plugin" in mapping:
        plugin_claim = True

    # 0b.2 Heuristic: In-Place Unrolling
    # If no mapping was found but method ends in '_' (e.g. x.add_), try to unroll it.
    if not plugin_claim and func_name and func_name.endswith("_") and not func_name.startswith("__"):
      hook = get_hook("unroll_inplace_ops")
      if hook:
        new_node = hook(updated, self.ctx)
        if new_node != updated:
          self._log_diff("In-place Unroll (Heuristic)", updated, new_node)
          # Adopt changes and update name resolution for subsequent passes
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
          self._log_diff("Lifecycle Strip", original, result_node)
          return result_node

      if method_name in warn_set:
        if isinstance(updated.func, cst.Attribute):
          self._report_warning(f"Ignored model state method '.{method_name}()'.")
          result_node = updated.func.value
          self._log_diff("Lifecycle Warn", original, result_node)
          return result_node

    # 1. Stateful Object Usage
    if func_name and self._is_stateful(func_name):
      fw_config = self.semantics.get_framework_config(self.target_fw)
      stateful_spec = fw_config.get("stateful_call")
      if stateful_spec:
        result_node = self._rewrite_stateful_call(updated, func_name, stateful_spec)
        self._log_diff("State Mechanism", original, result_node)
        return result_node

    if not func_name:
      # Fix: Prevent strict mode failure on super() calls
      if self._is_super_call(original):
        return updated

      if self.strict_mode:
        self._report_failure("Could not resolve function name")
      return updated

    # 2. Standard API Rewrite Setup
    mapping = self._get_mapping(func_name)
    if not mapping:
      # --- TRACE: Detailed Decision Log ---
      if not self._is_builtin(func_name):
        get_tracer().log_inspection(node_str=func_name, outcome="Skipped", detail="No Entry in Semantics Knowledge Base")
      return updated

    lookup = self.semantics.get_definition(func_name)
    if not lookup:
      return updated

    abstract_id, details = lookup

    # 2a. Infix / Prefix Transformation
    if mapping.get("transformation_type") == "infix":
      try:
        norm_args = self._normalize_arguments(original, updated, details, mapping)
        result_node = self._rewrite_as_infix(
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
        result_node = self._rewrite_as_inline_lambda(mapping["api"], norm_args)
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

    # 2d. Standard Function Rewrite
    else:
      try:
        norm_args = self._normalize_arguments(original, updated, details, mapping)
        new_func = self._create_name_node(mapping["api"])
        result_node = updated.with_changes(func=new_func, args=norm_args)
      except ValueError:
        self._report_failure("Argument normalization failed")
        return updated

    # 3. Output Normalization (Post-Processing)
    if "output_adapter" in mapping and mapping["output_adapter"]:
      try:
        result_node = self._apply_output_adapter(result_node, mapping["output_adapter"])
      except Exception as e:
        self._report_failure(f"Output adapter failed: {e}")
        return updated

    # 4. Class Construction Logic (Logic 4 implementation)
    if self._signature_stack and self._signature_stack[-1].is_init and self._signature_stack[-1].is_module_method:
      origins = getattr(self.semantics, "_key_origins", {})
      tier = origins.get(abstract_id)
      traits = self._get_target_traits()

      # Determine if this operation is a Neural Candidate
      # We check Semantic Tier (Preferred) OR explicit trait match (Fallback for Extras/Snapshots)
      is_neural_candidate = tier == SemanticTier.NEURAL.value

      # Fallback: If traits explicitly strip magic args (e.g. rngs),
      # we should check if they exist here even if categorization failed (e.g. op loaded as Extra)
      force_strip = False
      for bad_arg in traits.strip_magic_args:
        # Check if the generated call contains this arg as a keyword
        if isinstance(result_node, cst.Call):
          for arg in result_node.args:
            if arg.keyword and arg.keyword.value == bad_arg:
              force_strip = True
              break

      # Process Injection or Stripping
      if is_neural_candidate or force_strip:
        # Injection: If target requires 'rngs' (like Flax NNX) and defines it in magic args
        if any(name == "rngs" for name, _ in traits.inject_magic_args):
          if isinstance(result_node, cst.Call):
            result_node = self._inject_rngs_kwarg(result_node)

        # Stripping: If target explicitly strips 'rngs' (like Torch)
        elif "rngs" in traits.strip_magic_args:
          if isinstance(result_node, cst.Call):
            result_node = self._strip_kwarg(result_node, "rngs")

    self._log_diff(f"Operation ({abstract_id})", original, result_node)

    return result_node

  def _is_builtin(self, name: str) -> bool:
    """Avoid spamming logs for standard python builtins unless mapped."""
    return name in {"print", "len", "range", "super", "enumerate", "zip", "int", "float", "str"}

  def _log_diff(self, label: str, original: cst.CSTNode, modified: cst.CSTNode) -> None:
    """Helper to compute diff and log if changed."""
    src_before, src_after, is_changed = diff_nodes(original, modified)
    if is_changed:
      get_tracer().log_mutation(label, src_before, src_after)

  def _apply_output_adapter(self, inner_node: cst.CSTNode, adapter_str: str) -> cst.Call:
    """
    Wraps a node with a lambda adapter to normalize output.
    Transform: `node` -> `(adapter_str)(node)`
    """
    try:
      lambda_node = cst.parse_expression(adapter_str)
      parenthesized = lambda_node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
      return cst.Call(func=parenthesized, args=[cst.Arg(value=inner_node)])
    except cst.ParserSyntaxError:
      raise ValueError(f"Invalid syntax in output_adapter: {adapter_str}")

  def _rewrite_as_inline_lambda(self, lambda_str: str, args: list[cst.Arg]) -> cst.Call:
    """
    Wraps arguments in an Immediately Invoked Lambda Expression (IIFE).
    """
    try:
      parsed_expr = cst.parse_expression(lambda_str)
      parenthesized_lambda = parsed_expr.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
      return cst.Call(func=parenthesized_lambda, args=args)
    except cst.ParserSyntaxError:
      raise ValueError(f"Invalid lambda syntax in semantics: {lambda_str}")

  def _is_functional_apply(self, node: cst.Call) -> bool:
    """
    Detects if a call node matches the `obj.apply` pattern used in Flax Linen.
    """
    if isinstance(node.func, cst.Attribute):
      if node.func.attr.value == "apply":
        return True
    return False

  def _inject_rngs_kwarg(self, node: cst.Call) -> cst.Call:
    """Injects `rngs=rngs` into a constructor call."""
    # Duplicate check
    for arg in node.args:
      if arg.keyword and arg.keyword.value == "rngs":
        return node

    new_args = list(node.args)

    # Ensure comma on previous arg
    if len(new_args) > 0:
      last = new_args[-1]
      # Always force whitespace after the comma, replacing potentially dense trailing commas
      new_args[-1] = last.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    new_args.append(
      cst.Arg(
        keyword=cst.Name("rngs"),
        value=cst.Name("rngs"),
        equal=cst.AssignEqual(
          whitespace_before=cst.SimpleWhitespace(""),
          whitespace_after=cst.SimpleWhitespace(""),
        ),
      )
    )
    return node.with_changes(args=new_args)

  def _strip_kwarg(self, node: cst.Call, kw_name: str) -> cst.Call:
    """Removes a keyword argument from a call node."""
    filtered = []
    for arg in node.args:
      if arg.keyword and arg.keyword.value == kw_name:
        continue
      filtered.append(arg)

    # Clean trailing comma on new last arg
    if filtered and filtered[-1].comma != cst.MaybeSentinel.DEFAULT:
      last = filtered[-1]
      filtered[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return node.with_changes(args=filtered)

  def _rewrite_stateful_call(self, node: cst.Call, instance_name: str, config: Dict[str, str]) -> cst.Call:
    """Rewrites a call to a stateful object (Functional patterns only)."""
    new_args = list(node.args)
    target_arg_name = config.get("prepend_arg", "variables")

    if self._signature_stack:
      sig_ctx = self._signature_stack[-1]
      if target_arg_name not in sig_ctx.existing_args:
        found = any(n == target_arg_name for n, _ in sig_ctx.injected_args)
        if not found:
          sig_ctx.injected_args.append((target_arg_name, None))
          self._report_warning(f"Injected missing state argument '{target_arg_name}' into signature.")

    injected_arg = cst.Arg(
      value=cst.Name(target_arg_name),
      comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
    )
    new_args.insert(0, injected_arg)

    method_name = config.get("method")
    if method_name:
      new_func = cst.Attribute(
        value=self._create_dotted_name(instance_name),
        attr=cst.Name(method_name),
      )
    else:
      new_func = node.func

    return node.with_changes(func=new_func, args=new_args)

  def _is_super_call(self, node: cst.Call) -> bool:
    """Helper to identify direct super() usage or super().__init__()."""
    if isinstance(node.func, cst.Attribute):
      # Case: super().method()
      receiver = node.func.value
      if isinstance(receiver, cst.Call) and isinstance(receiver.func, cst.Name):
        if receiver.func.value == "super":
          return True
    elif isinstance(node.func, cst.Name):
      # Case: super()
      if node.func.value == "super":
        return True
    return False
