"""
Call Rewriting Logic.

Handles function invocation usage (`leave_Call`) and Assignment Unwrapping.
Includes functionality for:
- Lifecycle method stripping (e.g. `.to()`, `.cpu()`).
- Functional -> OOP Unwrapping (removing `.apply` and tuple unpacking).
- Stateful object management.
- Standard API pivots.
- Plugin dispatch.
- Constructor logic injection (RNG management).
- **Output Normalization**: Adapting return values to match the Abstract Standard.
"""

from typing import Union, Set, Dict
import libcst as cst

from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.core.rewriter.normalization import NormalizationMixin


class CallMixin(NormalizationMixin, BaseRewriter):
  """
  Mixin for transforming Call nodes and unpacking Assignments.
  """

  # Methods to strip completely (return receiver object)
  _LIFECYCLE_STRIP: Set[str] = {
    "to",
    "cpu",
    "cuda",
    "detach",
    "clone",
    "requires_grad_",
    "share_memory_",
  }

  # Methods to strip but warn about state implications.
  _LIFECYCLE_WARN: Set[str] = {
    "eval",
    "train",
    "half",
    "float",
    "double",
    "type",
  }

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """
    Handles assignment unwrapping for Functional -> OOP transitions.

    Scenario: `y, updates = layer.apply(vars, x)`
    Target:   `y = layer(x)` (NNX/Torch style)

    If we detected that a call was rewritten from a functional pattern
    returning multiple values (outputs + state) to an OOP pattern
    returning a single value, we must also simplify the assignment targets.
    """
    # 1. Check if we are assigning from a Call
    if not isinstance(original_node.value, cst.Call):
      return super().leave_Assign(original_node, updated_node)

    # 2. Check if the original call was '.apply' (The functional signature)
    # We look at the ORIGINAL node to see what the source code intended.
    if self._is_functional_apply(original_node.value):
      # Check if we have multiple targets (y, updates = ...)
      # cst.Assign.targets contains a list of AssignTarget.
      # Usually one AssignTarget which wraps a Tuple/List if destructuring.
      if len(updated_node.targets) == 1:
        target = updated_node.targets[0].target
        if isinstance(target, (cst.Tuple, cst.List)):
          # Destructuring detected: y, updates = ...
          # We assume index 0 is the prediction/output we care about in OOP.
          elements = target.elements
          if len(elements) > 0:
            primary_target = elements[0].value

            # Create new single target
            new_target = cst.AssignTarget(target=primary_target)
            return updated_node.with_changes(targets=[new_target])

    # Delegate to MRO (likely AttributeMixin)
    return super().leave_Assign(original_node, updated_node)

  def leave_Call(
    self, original: cst.Call, updated: cst.Call
  ) -> Union[cst.Call, cst.BinaryOperation, cst.UnaryOperation, cst.CSTNode]:
    """Rewrites function calls."""
    func_name = self._get_qualified_name(original.func)

    # 0a. Functional 'apply' unwrapping (Linen -> NNX/Torch)
    # Check against pure .apply calls on objects
    if self._is_functional_apply(original):
      # Transform: obj.apply(vars, x) -> obj(x)
      # 1. Remove '.apply' from function name
      # original.func is Attribute(value=obj, attr='apply')
      # new func is just 'obj' (the value)
      if isinstance(updated.func, cst.Attribute):
        receiver = updated.func.value

        # 2. Remove first argument 'vars' (state dictionary)
        # Arguments: [vars, x, y...] -> [x, y...]
        new_args = updated.args[1:] if len(updated.args) > 0 else []

        return updated.with_changes(func=receiver, args=new_args)

    # 0b. Check for Plugin First! (Override Lifecycle logic)
    plugin_claim = False
    if func_name:
      mapping = self._get_mapping(func_name)
      if mapping and "requires_plugin" in mapping:
        plugin_claim = True

    # 0c. Lifecycle Method Handling
    if not plugin_claim and isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
      method_name = original.func.attr.value

      # Case A: Strip (x.to() -> x)
      if method_name in self._LIFECYCLE_STRIP:
        if isinstance(updated.func, cst.Attribute):
          self._report_warning(f"Stripped framework-specific lifecycle method '.{method_name}()'.")
          return updated.func.value

      # Case B: Warn/Stub
      if method_name in self._LIFECYCLE_WARN:
        if isinstance(updated.func, cst.Attribute):
          self._report_warning(f"Ignored model state method '.{method_name}()'.")
          return updated.func.value

    # 1. Stateful Object Usage
    # If converting TO functional (not relevant for NNX target, but kept for logic safety)
    if func_name and self._is_stateful(func_name):
      fw_config = self.semantics.get_framework_config(self.target_fw)
      stateful_spec = fw_config.get("stateful_call")
      if stateful_spec:
        return self._rewrite_stateful_call(updated, func_name, stateful_spec)

    if not func_name:
      if self.strict_mode:
        self._report_failure("Could not resolve function name")
      return updated

    # 2. Standard API Rewrite Setup
    mapping = self._get_mapping(func_name)
    if not mapping:
      return updated

    lookup = self.semantics.get_definition(func_name)
    if not lookup:
      return updated

    abstract_id, details = lookup
    result_node = updated

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
        return updated  # Abort normalization

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
    # Check if target framework returns mismatched output (e.g. extra tuple args)
    if "output_adapter" in mapping and mapping["output_adapter"]:
      try:
        result_node = self._apply_output_adapter(result_node, mapping["output_adapter"])
      except Exception as e:
        self._report_failure(f"Output adapter failed: {e}")
        return updated

    # 4. Class Construction Logic (init/call within init)
    if self._signature_stack and self._signature_stack[-1].is_init and self._signature_stack[-1].is_module_method:
      origins = getattr(self.semantics, "_key_origins", {})
      tier = origins.get(abstract_id)
      if tier == SemanticTier.NEURAL.value:
        # Note: We must check if result_node is still a Call to append kwargs.
        # If it became an Expression (via infix or plugin), we skip.
        if isinstance(result_node, cst.Call):
          if self.target_fw == "jax":
            result_node = self._inject_rngs_kwarg(result_node)
          elif self.target_fw == "torch":
            result_node = self._strip_kwarg(result_node, "rngs")

    return result_node

  def _apply_output_adapter(self, inner_node: cst.CSTNode, adapter_str: str) -> cst.Call:
    """
    Wraps a node with a lambda adapter to normalize output.
    Transform: `node` -> `(adapter_str)(node)`
    Example: `max(x)` -> `(lambda x: x[0])(max(x))`
    """
    try:
      # Parse the lambda expression
      # "lambda x: x[0]" -> Lambda Node
      lambda_node = cst.parse_expression(adapter_str)

      # Wrap lambda in parens for immediate invocation: (lambda...)(arg)
      parenthesized = lambda_node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])

      # Construct wrapper Call
      # Argument passed is determining by the inner_node expression
      # We assume inner_node is an Expression (Call, Name, etc.)
      return cst.Call(func=parenthesized, args=[cst.Arg(value=inner_node)])
    except cst.ParserSyntaxError:
      raise ValueError(f"Invalid syntax in output_adapter: {adapter_str}")

  def _rewrite_as_inline_lambda(self, lambda_str: str, args: list[cst.Arg]) -> cst.Call:
    """
    Wraps arguments in an Immediately Invoked Lambda Expression (IIFE).

    Transforms:
        `api(args)` -> `(lambda x: ...)(args)`

    Args:
        lambda_str: The lambda definition code (e.g. "lambda x: hasattr(x, '__array__')").
        args: The normalized list of arguments to pass to the lambda.

    Returns:
        A CST Call node where the function is a parenthesized lambda expression.
    """
    try:
      # Parse the lambda string into a CST node
      # We parse it as an expression to get the Lambda node
      parsed_expr = cst.parse_expression(lambda_str)

      # Wrap it in parentheses to ensure valid calling syntax: (lambda...)(args)
      parenthesized_lambda = parsed_expr.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])

      return cst.Call(func=parenthesized_lambda, args=args)
    except cst.ParserSyntaxError:
      raise ValueError(f"Invalid lambda syntax in semantics: {lambda_str}")

  def _is_functional_apply(self, node: cst.Call) -> bool:
    """
    Detects if a call node matches the `obj.apply` pattern used in Flax Linen.
    We check if the attribute name is 'apply'.
    """
    if isinstance(node.func, cst.Attribute):
      if node.func.attr.value == "apply":
        return True
    return False

  def _inject_rngs_kwarg(self, node: cst.Call) -> cst.Call:
    """Injects `rngs=rngs` into a constructor call."""
    for arg in node.args:
      if arg.keyword and arg.keyword.value == "rngs":
        return node
    new_args = list(node.args)
    if len(new_args) > 0:
      last = new_args[-1]
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
