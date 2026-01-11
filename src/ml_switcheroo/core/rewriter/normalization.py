"""
Normalization Helpers for AST Rewriting (The Argument Pivot).

Provides the `NormalizationMixin`, handling argument pivoting, renaming,
and default injection. Uses `ApiStage` context for lookups.
"""

import libcst as cst
from typing import List, Dict, Any, TYPE_CHECKING

from ml_switcheroo.core.rewriter.normalization_utils import (
  convert_value_to_cst,
  extract_primitive_key,
)

if TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.calls.mixer import ApiStage


class NormalizationMixin:
  """
  Mixin class providing argument normalization logic.
  Expects `self.context` from ApiStage.
  """

  def _is_module_alias(self: "ApiStage", node: cst.CSTNode) -> bool:
    """
    Determines if a node represents a known framework module alias.
    """
    name = self._cst_to_string(node)
    if not name:
      return False

    # 1. Check known aliases map (from context)
    if name in self.context.alias_map:
      return True

    # 2. Check Dynamic Registry
    known_roots = set()

    # A. From Configuration
    if self.config:
      known_roots.add(self.config.source_framework)
      known_roots.add(self.config.target_framework)
      if self.config.source_flavour:
        known_roots.add(self.config.source_flavour.split(".")[0])

    # B. From Semantics Manager
    if self.context.semantics:
      configs = getattr(self.context.semantics, "framework_configs", {})
      for fw_key, conf in configs.items():
        known_roots.add(fw_key)
        alias_conf = conf.get("alias")
        if alias_conf and isinstance(alias_conf, dict):
          mod = alias_conf.get("module")
          if mod:
            known_roots.add(mod.split(".")[0])

      import_data = getattr(self.context.semantics, "import_data", {})
      for mod_path in import_data.keys():
        known_roots.add(mod_path.split(".")[0])

    root = name.split(".")[0]
    return root in known_roots

  def _normalize_arguments(
    self: "ApiStage",
    original_node: cst.Call,
    updated_node: cst.Call,
    op_details: Dict[str, Any],
    target_impl: Dict[str, Any],
  ) -> List[cst.Arg]:
    """
    Normalizes arguments from the source call to the target signature via the Pivot.
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
          if "default" in item:
            defaults_map[name] = item["default"]
      elif isinstance(item, (list, tuple)):
        std_args_order.append(item[0])
      else:
        std_args_order.append(item)

    # 2. Prepare Mapping Dictionaries
    source_variant = op_details.get("variants", {}).get(self.source_fw, {})
    if not source_variant:
      source_variant = {}

    source_arg_map = source_variant.get("args", {}) or {}
    target_arg_map = target_impl.get("args", {}) or {}
    target_val_map = target_impl.get("arg_values", {}) or {}
    pack_target_kw = target_impl.get("pack_to_tuple")
    pack_as_type = target_impl.get("pack_as", "Tuple")
    target_inject_map = target_impl.get("inject_args", {}) or {}

    lib_to_std = {v: k for k, v in source_arg_map.items()}

    found_args: Dict[str, cst.Arg] = {}
    extra_args: List[cst.Arg] = []
    variadic_buffer: List[cst.Arg] = []

    # 3. Method-to-Function Receiver Injection
    is_method_call = isinstance(original_node.func, cst.Attribute)
    receiver_injected = False

    if is_method_call and self._is_module_alias(original_node.func.value):
      is_method_call = False

    if is_method_call:
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
            found_args[first_std_arg] = cst.Arg(value=rec)
            receiver_injected = True
      else:
        if isinstance(original_node.func, cst.Attribute):
          extra_args.append(cst.Arg(value=original_node.func.value))

    # 4. Process Arguments from Call
    pos_idx = 1 if receiver_injected else 0
    packing_mode = False

    for i, upd_arg in enumerate(updated_node.args):
      if i < len(original_node.args):
        orig_arg = original_node.args[i]
      else:
        orig_arg = upd_arg

      if not orig_arg.keyword:
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
        k_name = orig_arg.keyword.value
        std_name = lib_to_std.get(k_name, k_name)
        if std_name in std_args_order:
          found_args[std_name] = upd_arg
        else:
          extra_args.append(upd_arg)

    # 5. Handle Packing Buffer
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

    # 6. Construct New List
    new_args_list: List[cst.Arg] = []

    for std_name in std_args_order:
      if std_name not in found_args and std_name in defaults_map:
        try:
          default_val = defaults_map[std_name]
          lit_val_node = convert_value_to_cst(default_val)
          found_args[std_name] = cst.Arg(
            keyword=cst.Name(std_name),
            value=lit_val_node,
            equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
          )
        except Exception:
          pass

      if std_name in found_args:
        current_arg = found_args[std_name]

        if std_name == variadic_arg_name and pack_target_kw:
          new_args_list.append(current_arg)
          continue

        # Check if argument is explicitly dropped
        # Logic: If arg is mapped to None in target_arg_map, we skip it.
        # get(key, default) returns None if key found and value is None.
        tg_alias = target_arg_map.get(std_name, std_name)

        if tg_alias is None:
          continue

        final_val_node = current_arg.value

        if target_val_map and std_name in target_val_map:
          val_options = target_val_map[std_name]
          raw_key = extract_primitive_key(current_arg.value)
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
          if final_val_node is not current_arg.value:
            new_arg = current_arg.with_changes(value=final_val_node)
            new_args_list.append(new_arg)
          else:
            new_args_list.append(current_arg)

    # Append extra arguments
    new_args_list.extend(extra_args)

    # 7. Inject Additional Arguments
    if target_inject_map:
      for arg_name, arg_val in target_inject_map.items():
        if any(a.keyword and a.keyword.value == arg_name for a in new_args_list):
          continue

        val_node = convert_value_to_cst(arg_val)

        if len(new_args_list) > 0 and new_args_list[-1].comma == cst.MaybeSentinel.DEFAULT:
          new_args_list[-1] = new_args_list[-1].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

        injected_arg = cst.Arg(
          keyword=cst.Name(arg_name),
          value=val_node,
          equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        )
        new_args_list.append(injected_arg)

    for i in range(len(new_args_list) - 1):
      new_args_list[i] = new_args_list[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    if new_args_list:
      new_args_list[-1] = new_args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return new_args_list
