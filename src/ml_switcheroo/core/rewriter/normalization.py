"""
Normalization Helpers for AST Rewriting (The Argument Pivot).

This module provides the `NormalizationMixin`, a component of the `PivotRewriter`
responsible for reshaping function arguments and call structures during translation.

Updates:
- Supports argument value mapping (Enums) via `arg_values`.
- Supports variadic argument packing via `pack_to_tuple`.
"""

from typing import List, Dict, Any, Union, Optional
import libcst as cst
from ml_switcheroo.core.rewriter.base import BaseRewriter


class NormalizationMixin(BaseRewriter):
  """
  Mixin class providing argument normalization and operator transformation logic.
  """

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """
    Determines if a node represents a known framework module alias.
    Used to prevent injecting the module itself as 'self' argument.

    Handles both simple names ('torch') and dotted paths ('flax.nnx').
    """
    # Resolve node to string representation (e.g. "torch.nn")
    name = self._cst_to_string(node)
    if not name:
      return False

    # 1. Check known aliases map (populated by BaseRewriter visits)
    if hasattr(self, "_alias_map") and name in self._alias_map:
      return True

    # 2. Check Common Roots
    root = name.split(".")[0]
    common_roots = {
      "torch",
      "jax",
      "tensorflow",
      "tf",
      "np",
      "numpy",
      "mx",
      "flax",
      "nn",
      "nnx",
      "F",
      "optim",
      "keras",
      "pl",
      "optax",
      "praxis",
      "orbax",
      "msgpack",
    }

    return root in common_roots

  def _extract_primitive_key(self, node: cst.BaseExpression) -> Optional[str]:
    """
    Extracts a string representation of a primitive AST node for key lookup.

    Args:
        node: The CST node to extract value from.

    Returns:
        String representation (e.g., "0", "mean", "True") or None if complex.
        Strings are returned without quotes.
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
    Supports renaming, reordering, value mapping, packing, and injection of new arguments.
    """
    # 1. Extract Standard Argument Order with metadata
    std_args_raw = op_details.get("std_args", [])
    std_args_order = []
    variadic_arg_name = None

    for item in std_args_raw:
      if isinstance(item, dict):
        name = item.get("name")
        if name:
          std_args_order.append(name)
          if item.get("is_variadic"):
            variadic_arg_name = name
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

    # Feature: Argument Packing (Tuple)
    pack_target_kw = target_impl.get("pack_to_tuple")

    # New: retrieve injected args for target
    target_inject_map = target_impl.get("inject_args", {})

    # Invert source map: {fw_name: std_name}
    lib_to_std = {v: k for k, v in source_arg_map.items()}

    found_args: Dict[str, cst.Arg] = {}
    extra_args: List[cst.Arg] = []
    variadic_buffer: List[cst.Arg] = []

    # 3. Method-to-Function Receiver Injection
    is_method_call = isinstance(original_node.func, cst.Attribute)
    receiver_injected = False

    # Distinguish `x.add()` from `torch.add()` or `flax.nnx.Linear()`
    if is_method_call:
      # If the value (left of dot) is a module alias, it's NOT a method call on a a tensor/object
      if self._is_module_alias(original_node.func.value):
        is_method_call = False

    if is_method_call and std_args_order:
      # Check if first arg is missing from call args
      first_std_arg = std_args_order[0]

      # Check if explicitly provided via kwarg
      arg_provided = False
      for arg in original_node.args:
        if arg.keyword:
          k_name = arg.keyword.value
          mapped = lib_to_std.get(k_name) or (k_name if k_name == first_std_arg else None)
          if mapped == first_std_arg:
            arg_provided = True
            break

      if not arg_provided:
        # Inject Receiver
        if isinstance(updated_node.func, cst.Attribute):
          rec = updated_node.func.value
        else:
          rec = original_node.func.value

        # Create arg without comma initially
        found_args[first_std_arg] = cst.Arg(value=rec)
        receiver_injected = True

    # 4. Process Arguments
    # Shift positional index if we injected arg 0
    pos_idx = 1 if receiver_injected else 0

    # Determine if packing should occur
    packing_mode = False

    for orig_arg, upd_arg in zip(original_node.args, updated_node.args):
      if not orig_arg.keyword:
        if packing_mode:
          # Continue collecting variadics
          variadic_buffer.append(upd_arg)
        elif pos_idx < len(std_args_order):
          std_name = std_args_order[pos_idx]

          # Check if this standard arg triggers packing
          if pack_target_kw and std_name == variadic_arg_name:
            packing_mode = True
            variadic_buffer.append(upd_arg)
            # std_name remains current for this buffer (we map it after loop)
          else:
            if std_name not in found_args:
              found_args[std_name] = upd_arg
            pos_idx += 1
        else:
          # Truly extra positional not accounted for by spec
          extra_args.append(upd_arg)
      else:
        k_name = orig_arg.keyword.value
        std_name = lib_to_std.get(k_name, k_name)

        if std_name in std_args_order:
          found_args[std_name] = upd_arg
        else:
          extra_args.append(upd_arg)

    # 5. Handle Packing Buffer
    if packing_mode and variadic_arg_name and pack_target_kw:
      # Create Tuple
      elements = []
      for arg in variadic_buffer:
        elements.append(cst.Element(value=arg.value, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))

      # Clean trailing comma for tuples if desired, though valid in Python
      if elements:
        elements[-1] = elements[-1].with_changes(
          comma=cst.MaybeSentinel.DEFAULT if len(elements) > 1 else cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
        )

      packed_tuple = cst.Tuple(elements=elements)

      # Create keyword argument for target
      packed_arg = cst.Arg(
        keyword=cst.Name(pack_target_kw),
        value=packed_tuple,
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
      )
      found_args[variadic_arg_name] = packed_arg

    # 6. Construct New List
    new_args_list: List[cst.Arg] = []
    for std_name in std_args_order:
      if std_name in found_args:
        current_arg = found_args[std_name]

        # If this was the packed argument, it already has the correct target keyword.
        if std_name == variadic_arg_name and pack_target_kw:
          new_args_list.append(current_arg)
          continue

        tg_alias = target_arg_map.get(std_name, std_name)

        # Apply Value Mapping
        final_val_node = current_arg.value
        if target_val_map and std_name in target_val_map:
          val_options = target_val_map[std_name]
          raw_key = self._extract_primitive_key(current_arg.value)
          # Check if key matches config
          if raw_key is not None and str(raw_key) in val_options:
            target_code = val_options[str(raw_key)]
            try:
              final_val_node = cst.parse_expression(target_code)
            except cst.ParserSyntaxError:
              # Fallback to original if code string is invalid
              pass

        if current_arg.keyword:
          new_arg = current_arg.with_changes(
            keyword=cst.Name(tg_alias),
            value=final_val_node,
            equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
          )
          new_args_list.append(new_arg)
        else:
          # Positional arg, update value if changed
          if final_val_node is not current_arg.value:
            new_arg = current_arg.with_changes(value=final_val_node)
            new_args_list.append(new_arg)
          else:
            new_args_list.append(current_arg)

    new_args_list.extend(extra_args)

    # 7. Inject Additional Arguments
    if target_inject_map:
      for arg_name, arg_val in target_inject_map.items():
        # Convert literal to CST node
        val_node = self._convert_value_to_cst(arg_val)

        # Don't inject if argument is already present (e.g. user provided it)
        # Check against existing keywords in new_args_list
        if any(a.keyword and a.keyword.value == arg_name for a in new_args_list):
          continue

        # Ensure previous arg has comma
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

    # Ensure commas
    for i in range(len(new_args_list) - 1):
      if new_args_list[i].comma == cst.MaybeSentinel.DEFAULT:
        new_args_list[i] = new_args_list[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    # Clean last comma
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
    else:
      # Fallback for unexpected types
      return cst.SimpleString(f"'{str(val)}'")
