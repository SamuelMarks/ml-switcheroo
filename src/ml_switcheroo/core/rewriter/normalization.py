"""
Normalization Helpers for AST Rewriting (The Argument Pivot).

This module provides the `NormalizationMixin`, a component of the `PivotRewriter`
responsible for reshaping function arguments and call structures during translation.

Handles:
-   **Argument Renaming**: Maps source keywords to target keywords (e.g. `dim` -> `axis`).
-   **Method-to-Function Pivot**: Injects the receiver object `x.add()` -> `add(x)` if needed.
-   **Default Value Injection**: Injects values defined in ODL `default` field if arguments are missing.
-   **Value Normalization**: Transforms primitives and nested containers defined in ODL defaults into CST nodes.
-   **Argument Values Enum Mapping**: Maps string/int literals (e.g. `reduction='mean'` -> `mode='avg'`).
-   **Packing**: Handles variadic argument packing into containers (List/Tuple).
"""

import libcst as cst
from typing import List, Dict, Any, Union

from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.core.rewriter.normalization_utils import (
  convert_value_to_cst,
  extract_primitive_key,
)


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

    Checks purely syntactic presence in local imports or semantic registry.

    Args:
        node: The CST node to inspect (Name or Attribute).

    Returns:
        True if it corresponds to a known framework root alias.
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
      configs = getattr(self.semantics, "framework_configs", {})
      for fw_key, conf in configs.items():
        known_roots.add(fw_key)
        # Handle Pydantic model dump or dict access
        alias_conf = conf.get("alias")
        if alias_conf and isinstance(alias_conf, dict):
          mod = alias_conf.get("module")
          if mod:
            known_roots.add(mod.split(".")[0])

      import_data = getattr(self.semantics, "import_data", {})
      for mod_path in import_data.keys():
        known_roots.add(mod_path.split(".")[0])

    root = name.split(".")[0]
    return root in known_roots

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
    defaults_map: Dict[str, Any] = {}
    variadic_arg_name = None

    for item in std_args_raw:
      if isinstance(item, dict):
        name = item.get("name")
        if name:
          std_args_order.append(name)
          if item.get("is_variadic"):
            variadic_arg_name = name
          # Capture default value if present key exists (even if value is None)
          if "default" in item:
            defaults_map[name] = item["default"]

      elif isinstance(item, (list, tuple)):
        std_args_order.append(item[0])
      else:
        std_args_order.append(item)

    # 2. Prepare Mapping Dictionaries
    # Access safety: if variants dict doesn't have key, return empty
    source_variant = op_details.get("variants", {}).get(self.source_fw, {})
    if not source_variant:
      source_variant = {}

    source_arg_map = source_variant.get("args", {}) or {}
    target_arg_map = target_impl.get("args", {}) or {}
    target_val_map = target_impl.get("arg_values", {}) or {}
    pack_target_kw = target_impl.get("pack_to_tuple")
    pack_as_type = target_impl.get("pack_as", "Tuple")
    target_inject_map = target_impl.get("inject_args", {}) or {}

    # Invert source map: {fw_name: std_name}
    lib_to_std = {v: k for k, v in source_arg_map.items()}

    found_args: Dict[str, cst.Arg] = {}
    extra_args: List[cst.Arg] = []
    variadic_buffer: List[cst.Arg] = []

    # 3. Method-to-Function Receiver Injection
    is_method_call = isinstance(original_node.func, cst.Attribute)
    receiver_injected = False

    # If the receiver IS the module (e.g. torch.add), it's not a method call on data.
    if is_method_call and self._is_module_alias(original_node.func.value):
      is_method_call = False

    if is_method_call:
      if std_args_order:
        first_std_arg = std_args_order[0]
        # Check if arg already provided by keyword (e.g. F.linear(input=x))
        arg_provided = False
        for arg in original_node.args:
          if arg.keyword:
            k_name = arg.keyword.value
            mapped = lib_to_std.get(k_name) or (k_name if k_name == first_std_arg else None)
            if mapped == first_std_arg:
              arg_provided = True
              break

        # If not provided explicitly, inject the receiver (Attribute.value) as the first argument
        if not arg_provided:
          if isinstance(original_node.func, cst.Attribute):
            rec = original_node.func.value
            found_args[first_std_arg] = cst.Arg(value=rec)
            receiver_injected = True
      else:
        # Fallback: if no metadata, just preserve receiver as first extra arg?
        # Usually implies malformed semantics, but safe behavior is append.
        if isinstance(original_node.func, cst.Attribute):
          extra_args.append(cst.Arg(value=original_node.func.value))

    # 4. Process Arguments from Call
    pos_idx = 1 if receiver_injected else 0
    packing_mode = False

    # We use updated_node args because children might have been rewritten recursively
    for i, upd_arg in enumerate(updated_node.args):
      # Access corresponding original arg to check structure (positional/keyword)
      # Safe index check in case updated_node length mismatch (should be same)
      if i < len(original_node.args):
        orig_arg = original_node.args[i]
      else:
        orig_arg = upd_arg

      if not orig_arg.keyword:
        # Positional
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
        # Keyword
        k_name = orig_arg.keyword.value
        std_name = lib_to_std.get(k_name, k_name)
        if std_name in std_args_order:
          found_args[std_name] = upd_arg
        else:
          extra_args.append(upd_arg)

    # 5. Handle Packing Buffer (variadic arguments)
    if packing_mode and variadic_arg_name and pack_target_kw:
      elements = []
      for arg in variadic_buffer:
        elements.append(
          cst.Element(
            value=arg.value,
            comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
          )
        )

      is_list = pack_as_type == "List"
      if elements:
        # Formatting: Lists no trailing comma, Tuples with 1 element need comma
        trailing_comma = cst.MaybeSentinel.DEFAULT
        if not is_list and len(elements) == 1:
          trailing_comma = cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))

        elements[-1] = elements[-1].with_changes(comma=trailing_comma)

      container_node = cst.List(elements=elements) if is_list else cst.Tuple(elements=elements)

      found_args[variadic_arg_name] = cst.Arg(
        keyword=cst.Name(pack_target_kw),
        value=container_node,
        equal=cst.AssignEqual(
          whitespace_before=cst.SimpleWhitespace(""),
          whitespace_after=cst.SimpleWhitespace(""),
        ),
      )

    # 6. Construct New List (Reordering, Renaming, Injection)
    new_args_list: List[cst.Arg] = []

    for std_name in std_args_order:
      # --- Logical Injection: Default Value Handling ---
      if std_name not in found_args and std_name in defaults_map:
        try:
          default_val = defaults_map[std_name]
          # Convert python object to CST node
          lit_val_node = convert_value_to_cst(default_val)

          found_args[std_name] = cst.Arg(
            keyword=cst.Name(std_name),  # Will be renamed below
            value=lit_val_node,
            equal=cst.AssignEqual(
              whitespace_before=cst.SimpleWhitespace(""),
              whitespace_after=cst.SimpleWhitespace(""),
            ),
          )
        except Exception:
          # Ignore if conversion fails, rely on target default
          pass
      # -----------------------------------------------

      if std_name in found_args:
        current_arg = found_args[std_name]

        # If the arg was packed, we already handled it in found_args dict?
        # Actually variadics don't usually map 1:1 in order list unless
        # they are singular. The `variadic_arg_name` check earlier handles packing construction.
        # If pack_target_kw is active, we put the packed arg here.
        if std_name == variadic_arg_name and pack_target_kw:
          new_args_list.append(current_arg)
          continue

        tg_alias = target_arg_map.get(std_name, std_name)

        # Apply Value Mapping (Enum Translation)
        final_val_node = current_arg.value
        if target_val_map and std_name in target_val_map:
          val_options = target_val_map[std_name]
          # Try to get string representation of key
          raw_key = extract_primitive_key(current_arg.value)
          if raw_key is not None and str(raw_key) in val_options:
            target_code = val_options[str(raw_key)]
            try:
              final_val_node = cst.parse_expression(target_code)
            except cst.ParserSyntaxError:
              pass

        # Determine if keyword is required
        # Logic: If arg has keyword, keep/rename keyword.
        # If arg is positional, but we reordered it or injected it, we force keyword?
        # Current Logic: If it HAD keyword, use TG alias keyword.
        # If it didn't have keyword, and value changed, update value.
        # Note: Default injection args created above have keywords.

        should_use_keyword = current_arg.keyword is not None

        if should_use_keyword:
          new_arg = current_arg.with_changes(
            keyword=cst.Name(tg_alias),
            value=final_val_node,
            equal=cst.AssignEqual(
              whitespace_before=cst.SimpleWhitespace(""),
              whitespace_after=cst.SimpleWhitespace(""),
            ),
          )
          new_args_list.append(new_arg)
        else:
          if final_val_node is not current_arg.value:
            new_arg = current_arg.with_changes(value=final_val_node)
            new_args_list.append(new_arg)
          else:
            new_args_list.append(current_arg)

    # Append extra arguments
    new_args_list.extend(extra_args)

    # 7. Inject Additional Arguments (Target Specific)
    if target_inject_map:
      for arg_name, arg_val in target_inject_map.items():
        # Don't inject if already present
        if any(a.keyword and a.keyword.value == arg_name for a in new_args_list):
          continue

        val_node = convert_value_to_cst(arg_val)

        # Ensure previous arg has comma
        if len(new_args_list) > 0 and new_args_list[-1].comma == cst.MaybeSentinel.DEFAULT:
          new_args_list[-1] = new_args_list[-1].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

        injected_arg = cst.Arg(
          keyword=cst.Name(arg_name),
          value=val_node,
          equal=cst.AssignEqual(
            whitespace_before=cst.SimpleWhitespace(""),
            whitespace_after=cst.SimpleWhitespace(""),
          ),
        )
        new_args_list.append(injected_arg)

    # Formatting Fixes: Ensure commas everywhere except last
    for i in range(len(new_args_list) - 1):
      new_args_list[i] = new_args_list[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    if new_args_list:
      new_args_list[-1] = new_args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return new_args_list
