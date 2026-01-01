"""
Normalization Helpers for AST Rewriting (The Argument Pivot).

This module provides the `NormalizationMixin`, a component of the `PivotRewriter`
responsible for reshaping function arguments and call structures during translation.

Updates:

-   Supports argument value mapping (Enums) via `arg_values`.
-   Supports variadic argument packing via `pack_to_tuple`.
-   **Feature Update (Configurable Packing)**: Supports `pack_as="List"` for list-based APIs.
-   **Feature Update**: Injects ODL `default` values if argument missing in source.
-   **Feature Update**: Handles dynamic module aliasing checks.
"""

from typing import List, Dict, Any, Union, Optional
import libcst as cst
from ml_switcheroo.core.rewriter.base import BaseRewriter


class NormalizationMixin(BaseRewriter):
  """
  Mixin class providing argument normalization and operator transformation logic.

  Dependencies:
      Requires `self.semantics` (SemanticsManager) and `self.config` (RuntimeConfig)
      to be available on the instance (inherited from BaseRewriter).
  """

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """
    Determines if a node represents a known framework module alias (e.g. 'torch', 'jnp').
    Used to prevent injecting the module itself as 'self' argument during method-to-function conversion.

    Logic:
        1. Checks local file aliases.
        2. Checks configured source/target frameworks.
        3. Checks semantics registry for known roots.
    """
    name = self._cst_to_string(node)
    if not name:
      return False

    # 1. Check known aliases map (populated by BaseRewriter.visit_Import)
    if hasattr(self, "_alias_map") and name in self._alias_map:
      return True

    # 2. Check Dynamic Registry (Semantic Knowledge Base)
    known_roots = set()

    # A. From Configuration (Source/Target are definitely frameworks)
    if hasattr(self, "config") and self.config:
      known_roots.add(self.config.source_framework)
      known_roots.add(self.config.target_framework)
      if self.config.source_flavour:
        known_roots.add(self.config.source_flavour.split(".")[0])
      if self.config.target_flavour:
        known_roots.add(self.config.target_flavour.split(".")[0])

    # B. From Semantics Manager (Registered Adapters)
    if hasattr(self, "semantics") and self.semantics:
      # framework_configs keys are framework IDs (e.g. 'torch', 'flax_nnx')
      configs = getattr(self.semantics, "framework_configs", {})
      for fw_key, conf in configs.items():
        known_roots.add(fw_key)

        # Check alias config (e.g. module='jax.numpy', name='jnp')
        alias_conf = conf.get("alias")
        if alias_conf and isinstance(alias_conf, dict):
          mod = alias_conf.get("module")
          if mod:
            known_roots.add(mod.split(".")[0])

      # Check import_data keys (keys are module paths like 'torch.nn')
      import_data = getattr(self.semantics, "import_data", {})
      for mod_path in import_data.keys():
        known_roots.add(mod_path.split(".")[0])

    # Validation
    root = name.split(".")[0]
    return root in known_roots

  def _extract_primitive_key(self, node: cst.BaseExpression) -> Optional[str]:
    """
    Extracts a string representation of a primitive AST node for key lookup.
    Used for Enum value mapping.

    Args:
        node: The CST node to extract value from.

    Returns:
        String representation (e.g., "0", "mean", "True") or None if complex.
    """
    if isinstance(node, cst.SimpleString):
      # "mean" -> mean
      return node.value.strip("'").strip('"')
    elif isinstance(node, cst.Integer):
      return node.value
    elif isinstance(node, cst.Name):
      # True, False, None
      return node.value
    return None

  def _normalize_arguments(
    self,
    original_node: cst.Call,
    updated_node: cst.Call,
    op_details: Dict[str, Any],
    target_impl: Dict[str, Any],
  ) -> List[cst.Arg]:
    """
    Normalizes arguments from the source call to the target signature via the Pivot.
    Supports renaming, reordering, value mapping, packing, default injection, and extra args.

    Args:
        original_node: The original CST node (for inspecting keywords/positions).
        updated_node: The transformed CST node (containing rewritten children).
        op_details: The abstract operation definition from Semantics (Hub).
        target_impl: The implementation definition for the target framework (Spoke).

    Returns:
        A list of CST Arg nodes representing the new function call signature.
    """
    # 1. Extract Standard Argument Order with metadata
    std_args_raw = op_details.get("std_args", [])
    std_args_order = []
    defaults_map: Dict[str, str] = {}
    variadic_arg_name = None

    for item in std_args_raw:
      if isinstance(item, dict):
        name = item.get("name")
        if name:
          std_args_order.append(name)
          if item.get("is_variadic"):
            variadic_arg_name = name
          # Capture default value if present
          if item.get("default") is not None:
            defaults_map[name] = str(item.get("default"))

      elif isinstance(item, (list, tuple)):
        std_args_order.append(item[0])
      else:
        std_args_order.append(item)

    # 2. Prepare Mapping Dictionaries
    source_variant = op_details["variants"].get(self.source_fw, {})
    source_arg_map = source_variant.get("args", {})
    target_arg_map = target_impl.get("args", {})

    # Feature: Argument Value Mapping
    target_val_map = target_impl.get("arg_values", {})

    # Feature: Argument Packing (Tuple/List)
    pack_target_kw = target_impl.get("pack_to_tuple")
    pack_as_type = target_impl.get("pack_as", "Tuple")  # Default to Tuple

    # Feature: Argument Injection
    target_inject_map = target_impl.get("inject_args", {})

    # Invert source map: {fw_name: std_name}
    lib_to_std = {v: k for k, v in source_arg_map.items()}

    found_args: Dict[str, cst.Arg] = {}
    extra_args: List[cst.Arg] = []
    variadic_buffer: List[cst.Arg] = []

    # 3. Method-to-Function Receiver Injection
    is_method_call = isinstance(original_node.func, cst.Attribute)
    receiver_injected = False

    # Distinguish `x.add()` from `torch.add()`
    if is_method_call:
      if self._is_module_alias(original_node.func.value):
        is_method_call = False

    if is_method_call:
      # Case A: We have a standard mapping (e.g. method X corresponds to arg 'x')
      if std_args_order:
        first_std_arg = std_args_order[0]
        arg_provided = False
        for arg in original_node.args:
          if arg.keyword:
            k_name = arg.keyword.value
            mapped = lib_to_std.get(k_name) or (k_name if k_name == first_std_arg else None)
            if mapped == first_std_arg:
              arg_provided = True
              break

        if not arg_provided:
          if isinstance(original_node.func, cst.Attribute):
            rec = original_node.func.value
            # Create arg without comma initially
            found_args[first_std_arg] = cst.Arg(value=rec)
            receiver_injected = True

      # Case B: Safety Fallback for Methods with missing/empty spec
      else:
        if isinstance(original_node.func, cst.Attribute):
          rec = original_node.func.value
          extra_args.append(cst.Arg(value=rec))

    # 4. Process Arguments from Call
    pos_idx = 1 if receiver_injected else 0
    packing_mode = False

    for orig_arg, upd_arg in zip(original_node.args, updated_node.args):
      if not orig_arg.keyword:
        # Positional Argument handling
        if packing_mode:
          variadic_buffer.append(upd_arg)
        elif pos_idx < len(std_args_order):
          std_name = std_args_order[pos_idx]
          if pack_target_kw and std_name == variadic_arg_name:
            packing_mode = True
            variadic_buffer.append(upd_arg)
          else:
            if std_name not in found_args:
              found_args[std_name] = upd_arg
            pos_idx += 1
        else:
          extra_args.append(upd_arg)
      else:
        # Keyword Argument handling
        k_name = orig_arg.keyword.value
        std_name = lib_to_std.get(k_name, k_name)
        if std_name in std_args_order:
          found_args[std_name] = upd_arg
        else:
          extra_args.append(upd_arg)

    # 5. Handle Packing Buffer (for variadic args like *dims)
    if packing_mode and variadic_arg_name and pack_target_kw:
      elements = []
      for arg in variadic_buffer:
        elements.append(cst.Element(value=arg.value, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))

      # Use Tuple or List based on config and fix trailing comma logic
      is_list = pack_as_type == "List"

      if elements:
        # Determine trailing comma requirement for single element
        # List: [x] (No comma usually required/preferred for lists)
        # Tuple: (x,) (Comma required)
        trailing_comma = cst.MaybeSentinel.DEFAULT
        if not is_list and len(elements) == 1:
          trailing_comma = cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))

        elements[-1] = elements[-1].with_changes(comma=trailing_comma)

      if is_list:
        container_node = cst.List(elements=elements)
      else:
        container_node = cst.Tuple(elements=elements)

      packed_arg = cst.Arg(
        keyword=cst.Name(pack_target_kw),
        value=container_node,
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
      )
      found_args[variadic_arg_name] = packed_arg

    # 6. Construct New List (Applying Defaults, Renaming, Value Mapping)
    new_args_list: List[cst.Arg] = []

    for std_name in std_args_order:
      # --- Logic Injection: Default Value Handling ---
      if std_name not in found_args and std_name in defaults_map:
        # Argument unused in source, but Spec defines a default.
        # Must inject to ensure semantic equivalence in target framework.
        try:
          default_code = defaults_map[std_name]
          lit_val_node = cst.parse_expression(default_code)

          found_args[std_name] = cst.Arg(
            keyword=cst.Name(std_name),  # Assign standard name temporarily
            value=lit_val_node,
            equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
          )
        except Exception:
          # If default string is invalid python (unlikely from ODL), skip injection
          pass
      # -----------------------------------------------

      if std_name in found_args:
        current_arg = found_args[std_name]

        if std_name == variadic_arg_name and pack_target_kw:
          new_args_list.append(current_arg)
          continue

        tg_alias = target_arg_map.get(std_name, std_name)

        # Apply Value Mapping (Enums)
        final_val_node = current_arg.value
        if target_val_map and std_name in target_val_map:
          val_options = target_val_map[std_name]
          raw_key = self._extract_primitive_key(current_arg.value)
          if raw_key is not None and str(raw_key) in val_options:
            target_code = val_options[str(raw_key)]
            try:
              final_val_node = cst.parse_expression(target_code)
            except cst.ParserSyntaxError:
              pass

        should_use_keyword = current_arg.keyword is not None

        if should_use_keyword:
          new_arg = current_arg.with_changes(
            keyword=cst.Name(tg_alias),
            value=final_val_node,
            equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
          )
          new_args_list.append(new_arg)
        else:
          # Preserve positional style unless value changed significantly,
          # but logic allows updating value node regardless.
          if final_val_node is not current_arg.value:
            new_arg = current_arg.with_changes(value=final_val_node)
            new_args_list.append(new_arg)
          else:
            new_args_list.append(current_arg)

    new_args_list.extend(extra_args)

    # 7. Inject Additional Arguments (from Target Variant config)
    if target_inject_map:
      for arg_name, arg_val in target_inject_map.items():
        val_node = self._convert_value_to_cst(arg_val)
        # Don't overwrite if present
        if any(a.keyword and a.keyword.value == arg_name for a in new_args_list):
          continue

        # Format previous comma
        if len(new_args_list) > 0:
          last = new_args_list[-1]
          if last.comma == cst.MaybeSentinel.DEFAULT:
            new_args_list[-1] = last.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

        injected_arg = cst.Arg(
          keyword=cst.Name(arg_name),
          value=val_node,
          equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        )
        new_args_list.append(injected_arg)

    # Final cleanup: Ensure all internal items have commas
    for i in range(len(new_args_list) - 1):
      new_args_list[i] = new_args_list[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    # Ensure last item has no comma
    if new_args_list:
      new_args_list[-1] = new_args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return new_args_list

  def _convert_value_to_cst(self, val: Any) -> cst.BaseExpression:
    """Converts a python primitive to a CST node."""
    if isinstance(val, bool):
      return cst.Name("True") if val else cst.Name("False")
    elif isinstance(val, int):
      return cst.Integer(str(val))
    elif isinstance(val, float):
      return cst.Float(str(val))
    elif isinstance(val, str):
      return cst.SimpleString(f"'{val}'")
    return cst.SimpleString(f"'{str(val)}'")
