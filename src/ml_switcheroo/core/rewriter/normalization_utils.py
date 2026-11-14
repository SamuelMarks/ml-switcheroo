"""Normalization Utilities.

This module provides helper functions for converting between Python runtime values
and LibCST Abstract Syntax Tree nodes. It supports primitive types (int, float,
bool, str) and recursive container types (list, tuple, dict).
"""

import ast
import dataclasses
import json
import typing

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.core.dsl import ParameterDef
from ml_switcheroo.enums import RuntimeTensorOrValue


def extract_primitive_key(node: cst.BaseExpression) -> str | None:
  """Extract a string representation of a primitive AST node for Enum key lookup.

  Handles simple strings, integers, and simple names (identifiers).

  Args:
      node: The CST expression node.

  Returns:
      The string value (if literal) or variable name. Returns None
      if the node type is complex or unsupported.

  """
  if isinstance(node, cst.SimpleString):
    return node.value.strip("'").strip('"')
  if isinstance(node, (cst.Integer, cst.Name)):
    return node.value
  return None


CstConvertibleVal: typing.TypeAlias = JsonValue | tuple[JsonValue, ...] | complex | RuntimeTensorOrValue


def _convert_container_to_cst(
  val: typing.Sequence[CstConvertibleVal] | typing.Mapping[str, CstConvertibleVal],
) -> cst.BaseExpression:
  """Convert a Python ``list``, ``tuple``, or ``dict`` into a LibCST literal container expression.

  Args:
      val: Container value to serialize into CST nodes.

  Returns:
      ``cst.List``, ``cst.Tuple``, or ``cst.Dict`` expression node.
  """
  if isinstance(val, typing.Mapping):
    dict_elements = [
      cst.DictElement(
        key=convert_value_to_cst(k),
        value=convert_value_to_cst(v),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      )
      for k, v in val.items()
    ]
    if dict_elements:
      dict_elements[-1] = dict_elements[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
    return cst.Dict(elements=dict_elements)

  elements = [
    cst.Element(
      value=convert_value_to_cst(item),
      comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
    )
    for item in val
  ]
  if elements:
    elements[-1] = elements[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  return cst.List(elements=elements) if isinstance(val, list) else cst.Tuple(elements=elements)


def convert_value_to_cst(
  val: CstConvertibleVal,
) -> cst.BaseExpression:
  """Recursively converts a python value (primitive/container) to a CST literal expression node.

  Supported types:
  - Primitives: bool, int, float, str, None
  - Containers: list, tuple, dict

  Args:
      val: The python value to convert.

  Returns:
      The corresponding LibCST node.

  """
  if isinstance(val, str):
    try:
      parsed = ast.literal_eval(val)
      if not isinstance(parsed, str) or parsed != val:
        return convert_value_to_cst(parsed)
    except (ValueError, SyntaxError):
      pass

  if isinstance(val, (list, tuple, dict)):
    return _convert_container_to_cst(val)
  if isinstance(val, bool):
    return cst.Name("True") if val else cst.Name("False")
  if isinstance(val, int):
    return (
      cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer(str(-val))) if val < 0 else cst.Integer(str(val))
    )
  if isinstance(val, float):
    return cst.UnaryOperation(operator=cst.Minus(), expression=cst.Float(repr(-val))) if val < 0 else cst.Float(repr(val))
  if isinstance(val, str):
    return cst.SimpleString(json.dumps(val))
  if val is None:
    return cst.Name("None")
  return cst.SimpleString(repr(str(val)))


def _extract_std_arg_specs(
  std_args_raw: typing.Sequence[JsonValue | ParameterDef | tuple[str, str]],
) -> tuple[list[str], dict[str, JsonValue], str | None]:
  """Parse raw ``std_args`` declarations into ordered names, defaults, and variadic parameter name.

  Args:
      std_args_raw: List of raw standard argument declarations from operation details.

  Returns:
      Tuple of (std_args_order, defaults_map, variadic_arg_name).
  """
  std_args_order: list[str] = []
  defaults_map: dict[str, JsonValue] = {}
  variadic_arg_name: str | None = None

  for item in std_args_raw:
    if isinstance(item, dict):
      name = item.get("name")
      if isinstance(name, str) and name:
        std_args_order.append(name)
        if item.get("is_variadic"):
          variadic_arg_name = name
        if "default" in item:
          defaults_map[name] = item["default"]
    elif isinstance(item, (list, tuple)) and item:
      std_args_order.append(str(item[0]))
    else:
      std_args_order.append(str(item))

  return std_args_order, defaults_map, variadic_arg_name


def _inject_method_receiver(
  original_node: cst.Call,
  std_args_order: list[str],
  lib_to_std: dict[str, str],
  found_args: dict[str, cst.Arg],
  extra_args: list[cst.Arg],
  is_module_alias_fn: typing.Callable[[cst.BaseExpression], bool],
) -> bool:
  """Inject the method receiver expression as the first standard argument when calling an instance method.

  Args:
      original_node: Original LibCST Call node.
      std_args_order: Ordered list of standard argument names.
      lib_to_std: Mapping from source framework argument names to standard argument names.
      found_args: Mutable dictionary of bound standard arguments.
      extra_args: Mutable list of extra unmapped arguments.
      is_module_alias_fn: Predicate checking whether an expression is a module alias.

  Returns:
      True if the receiver expression was injected into ``found_args``, False otherwise.
  """
  is_method_call = isinstance(original_node.func, cst.Attribute)
  if isinstance(original_node.func, cst.Attribute) and is_module_alias_fn(original_node.func.value):
    is_method_call = False

  if not is_method_call:
    return False

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

    if not arg_provided and isinstance(original_node.func, cst.Attribute):
      found_args[first_std_arg] = cst.Arg(value=original_node.func.value)
      return True
  elif isinstance(original_node.func, cst.Attribute):
    extra_args.append(cst.Arg(value=original_node.func.value))

  return False


def _pack_variadic_args(
  variadic_buffer: list[cst.Arg],
  pack_target_kw: str,
  pack_as_type: str,
) -> cst.Arg:
  """Pack a sequence of variadic positional arguments into a Tuple or List keyword argument.

  Args:
      variadic_buffer: List of variadic CST Arg nodes to pack.
      pack_target_kw: Target keyword parameter name for the packed container.
      pack_as_type: Container type identifier ('List' or 'Tuple').

  Returns:
      Constructed LibCST Arg node containing the packed Tuple or List.
  """
  elements = [
    cst.Element(
      value=arg.value,
      comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
    )
    for arg in variadic_buffer
  ]

  is_list = pack_as_type == "List"
  trailing_comma: cst.MaybeSentinel | cst.Comma = cst.MaybeSentinel.DEFAULT
  if not is_list and len(elements) == 1:
    trailing_comma = cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
  elements[-1] = elements[-1].with_changes(comma=trailing_comma)

  container_node = cst.List(elements=elements) if is_list else cst.Tuple(elements=elements)

  return cst.Arg(
    keyword=cst.Name(pack_target_kw),
    value=container_node,
    equal=cst.AssignEqual(
      whitespace_before=cst.SimpleWhitespace(""),
      whitespace_after=cst.SimpleWhitespace(""),
    ),
  )


def _resolve_target_arg_value(
  std_name: str,
  current_arg: cst.Arg,
  target_val_map: typing.Mapping[str, RuntimeTensorOrValue],
) -> cst.BaseExpression:
  """Resolve enum value mappings or constant value overrides for a standard argument.

  Args:
      std_name: Standard argument name.
      current_arg: Current CST Arg node bound to ``std_name``.
      target_val_map: Target value mapping dictionary from ``target_impl['arg_values']``.

  Returns:
      Resolved LibCST expression node for the argument value.
  """
  final_val_node = current_arg.value
  if not target_val_map or std_name not in target_val_map:
    return final_val_node

  val_options = target_val_map[std_name]
  if isinstance(val_options, dict):
    raw_key = extract_primitive_key(current_arg.value)
    if raw_key is not None and str(raw_key) in val_options:
      final_val_node = cst.parse_expression(str(val_options[str(raw_key)]))
  else:
    target_code = val_options
    if isinstance(target_code, str):
      try:
        final_val_node = cst.parse_expression(target_code)
      except cst.ParserSyntaxError:
        final_val_node = cst.SimpleString(f"'{target_code}'")
    else:
      final_val_node = convert_value_to_cst(target_code)

  return final_val_node


def _inject_trailing_args(
  new_args_list: list[cst.Arg],
  std_args_order: list[str],
  target_inject_map: typing.Mapping[str, RuntimeTensorOrValue],
  target_val_map: typing.Mapping[str, RuntimeTensorOrValue],
) -> None:
  """Append target-specific injected keyword arguments to ``new_args_list``.

  Args:
      new_args_list: Mutable list of normalized CST Arg nodes.
      std_args_order: Ordered list of standard argument names.
      target_inject_map: Target ``inject_args`` dictionary.
      target_val_map: Target ``arg_values`` dictionary.
  """
  injections: dict[str, RuntimeTensorOrValue] = dict(target_inject_map)
  if target_val_map:
    injections.update({k: v for k, v in target_val_map.items() if not isinstance(v, dict) and k not in std_args_order})

  for arg_name, arg_val in injections.items():
    val_node = cst.parse_expression(arg_val) if isinstance(arg_val, str) else convert_value_to_cst(arg_val)

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


@dataclasses.dataclass(frozen=True)
class _NormalizationMaps:
  """Parsed source and target argument configuration maps for ``normalize_arguments``.

  Attributes:
      lib_to_std: Source parameter name to standard parameter name mapping.
      target_arg_map: Standard parameter name to target parameter name (or ``None`` to drop).
      target_val_map: Target constant or enum value substitution map.
      pack_target_kw: Target keyword parameter name when variadic packing is enabled.
      pack_as_type: Target container kind (``'Tuple'`` or ``'List'``) for variadic packing.
      target_inject_map: Additional trailing keyword arguments to inject.
      kwargs_map: Optional filtering/renaming map for unmapped extra keyword arguments.
      should_inject_defaults: Whether standard default values should be materialized.
  """

  lib_to_std: dict[str, str]
  target_arg_map: dict[str, str | None]
  target_val_map: dict[str, JsonValue]
  pack_target_kw: str | None
  pack_as_type: str
  target_inject_map: dict[str, JsonValue]
  kwargs_map: dict[str, str | None]
  should_inject_defaults: bool


def _build_normalization_maps(
  op_details: typing.Mapping[str, RuntimeTensorOrValue],
  target_impl: typing.Mapping[str, RuntimeTensorOrValue],
  source_framework: str,
) -> _NormalizationMaps:
  """Extract typed argument mapping configurations from ``op_details`` and ``target_impl``.

  Args:
      op_details: Standard operation definition mapping.
      target_impl: Target framework variant configuration mapping.
      source_framework: Source framework key.

  Returns:
      Populated ``_NormalizationMaps`` configuration bundle.
  """
  raw_variants = op_details.get("variants", {})
  source_variant = raw_variants.get(source_framework, {}) if isinstance(raw_variants, dict) else {}
  raw_src_args = source_variant.get("args", {}) if isinstance(source_variant, dict) else {}
  source_arg_map = (
    {str(k): str(v) for k, v in raw_src_args.items() if v is not None} if isinstance(raw_src_args, dict) else {}
  )

  raw_tgt_args = target_impl.get("args", {})
  target_arg_map = (
    {str(k): (str(v) if v is not None else None) for k, v in raw_tgt_args.items()}
    if isinstance(raw_tgt_args, dict)
    else {}
  )
  raw_val_map = target_impl.get("arg_values", {})
  raw_pack_kw = target_impl.get("pack_to_tuple")
  raw_pack_as = target_impl.get("pack_as", "Tuple")
  raw_inject_map = target_impl.get("inject_args", {})
  raw_kwargs_map = target_impl.get("kwargs_map", {})

  return _NormalizationMaps(
    lib_to_std={v: k for k, v in source_arg_map.items()},
    target_arg_map=target_arg_map,
    target_val_map=dict(raw_val_map) if isinstance(raw_val_map, dict) else {},
    pack_target_kw=str(raw_pack_kw) if isinstance(raw_pack_kw, str) and raw_pack_kw else None,
    pack_as_type=str(raw_pack_as) if isinstance(raw_pack_as, str) else "Tuple",
    target_inject_map=dict(raw_inject_map) if isinstance(raw_inject_map, dict) else {},
    kwargs_map=(
      {str(k): (str(v) if v is not None else None) for k, v in raw_kwargs_map.items()}
      if isinstance(raw_kwargs_map, dict)
      else {}
    ),
    should_inject_defaults=bool(target_impl.get("inject_defaults", True)),
  )


def _bind_positional_and_keyword_args(
  args: typing.Sequence[cst.Arg],
  std_args_order: list[str],
  variadic_arg_name: str | None,
  maps: _NormalizationMaps,
  found_args: dict[str, cst.Arg],
  extra_args: list[cst.Arg],
  *,
  receiver_injected: bool,
) -> None:
  """Bind positional, keyword, and variadic call arguments into ``found_args`` and ``extra_args``.

  Args:
      args: Updated call argument sequence.
      std_args_order: Ordered standard parameter names.
      variadic_arg_name: Optional variadic standard parameter name.
      maps: Parsed normalization maps.
      found_args: Mutable bound standard argument dictionary.
      extra_args: Mutable unmapped argument list.
      receiver_injected: Whether ``std_args_order[0]`` was already bound to the method receiver.
  """
  pos_idx = 1 if receiver_injected else 0
  packing_mode = False
  variadic_buffer: list[cst.Arg] = []

  for upd_arg in args:
    if not upd_arg.keyword:
      if packing_mode:
        variadic_buffer.append(upd_arg)
      elif pos_idx < len(std_args_order):
        std_name = std_args_order[pos_idx]
        if maps.pack_target_kw and std_name == variadic_arg_name:
          packing_mode = True
          variadic_buffer.append(upd_arg)
        else:
          found_args.setdefault(std_name, upd_arg)
          pos_idx += 1
      else:
        extra_args.append(upd_arg)
    else:
      k_name = upd_arg.keyword.value
      std_name = maps.lib_to_std.get(k_name, k_name)
      if std_name in std_args_order:
        found_args[std_name] = upd_arg
      else:
        extra_args.append(upd_arg)

  if packing_mode and variadic_arg_name and maps.pack_target_kw:
    found_args[variadic_arg_name] = _pack_variadic_args(variadic_buffer, maps.pack_target_kw, maps.pack_as_type)


def _materialize_default_args(
  std_args_order: list[str],
  defaults_map: dict[str, JsonValue],
  found_args: dict[str, cst.Arg],
) -> None:
  """Populate missing standard arguments in ``found_args`` from ``defaults_map``.

  Args:
      std_args_order: Ordered standard parameter names.
      defaults_map: Default values keyed by standard parameter name.
      found_args: Mutable bound standard argument dictionary.
  """
  for std_name in std_args_order:
    if std_name not in found_args and std_name in defaults_map:
      try:
        lit_val_node = convert_value_to_cst(defaults_map[std_name])
        found_args[std_name] = cst.Arg(
          keyword=cst.Name(std_name),
          value=lit_val_node,
          equal=cst.AssignEqual(
            whitespace_before=cst.SimpleWhitespace(""),
            whitespace_after=cst.SimpleWhitespace(""),
          ),
        )
      except (
        cst.ParserSyntaxError,
        cst.CSTValidationError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RuntimeError,
      ):
        pass


def _emit_ordered_target_args(
  std_args_order: list[str],
  variadic_arg_name: str | None,
  found_args: dict[str, cst.Arg],
  maps: _NormalizationMaps,
) -> list[cst.Arg]:
  """Emit ordered ``cst.Arg`` nodes aligned and renamed for the target implementation.

  Args:
      std_args_order: Ordered standard parameter names.
      variadic_arg_name: Optional variadic standard parameter name.
      found_args: Bound standard arguments dictionary.
      maps: Parsed normalization maps.

  Returns:
      Ordered list of target ``cst.Arg`` nodes prior to extra/injected arguments.
  """
  emitted: list[cst.Arg] = []
  for std_name in std_args_order:
    if std_name not in found_args:
      continue
    current_arg = found_args[std_name]
    if std_name == variadic_arg_name and maps.pack_target_kw:
      emitted.append(current_arg)
      continue

    raw_alias = maps.target_arg_map.get(std_name, std_name)
    if raw_alias is None:
      continue
    tg_alias = str(raw_alias)
    final_val_node = _resolve_target_arg_value(std_name, current_arg, maps.target_val_map)

    if current_arg.keyword is not None:
      emitted.append(
        current_arg.with_changes(
          keyword=cst.Name(tg_alias),
          value=final_val_node,
          equal=cst.AssignEqual(
            whitespace_before=cst.SimpleWhitespace(""),
            whitespace_after=cst.SimpleWhitespace(""),
          ),
        )
      )
    elif final_val_node is not current_arg.value:
      emitted.append(current_arg.with_changes(value=final_val_node))
    else:
      emitted.append(current_arg)
  return emitted


def normalize_arguments(
  original_node: cst.Call,
  updated_node: cst.Call,
  op_details: typing.Mapping[str, RuntimeTensorOrValue],
  target_impl: typing.Mapping[str, RuntimeTensorOrValue],
  source_framework: str,
  is_module_alias_fn: typing.Callable[[cst.BaseExpression], bool],
) -> list[cst.Arg]:
  """Pivot arguments from source implementation -> Standard -> Target implementation.

  Handles renaming, reordering, and default injection.

  Args:
      original_node: The original call node from the source AST.
      updated_node: The updated call node being processed.
      op_details: A dictionary containing details of the standard operation, including
          standard arguments (`std_args`) and variants (`variants`).
      target_impl: A dictionary containing configuration details for the target
          implementation, including argument mappings (`args`), default values (`arg_values`),
          injected arguments (`inject_args`), and packing settings (`pack_to_tuple`).
      source_framework: The name of the source framework (e.g., 'pytorch', 'tensorflow', 'jax').
      is_module_alias_fn: A function or callable used to determine if a node represents
          a module alias or namespace.

  Returns:
      A list of LibCST Argument (`cst.Arg`) nodes aligned, named, and formatted
      for the target framework's implementation.
  """
  raw_std_args = op_details.get("std_args", [])
  std_args_seq = raw_std_args if isinstance(raw_std_args, (list, tuple)) else []
  std_args_order, defaults_map, variadic_arg_name = _extract_std_arg_specs(std_args_seq)
  maps = _build_normalization_maps(op_details, target_impl, source_framework)

  found_args: dict[str, cst.Arg] = {}
  extra_args: list[cst.Arg] = []

  receiver_injected = _inject_method_receiver(
    original_node,
    std_args_order,
    maps.lib_to_std,
    found_args,
    extra_args,
    is_module_alias_fn,
  )
  _bind_positional_and_keyword_args(
    updated_node.args,
    std_args_order,
    variadic_arg_name,
    maps,
    found_args,
    extra_args,
    receiver_injected=receiver_injected,
  )

  if maps.should_inject_defaults:
    _materialize_default_args(std_args_order, defaults_map, found_args)

  new_args_list = _emit_ordered_target_args(std_args_order, variadic_arg_name, found_args, maps)

  if maps.kwargs_map:
    extra_args = [
      arg
      for arg in extra_args
      if not (arg.keyword and arg.keyword.value in maps.kwargs_map and maps.kwargs_map[arg.keyword.value] is None)
    ]
  new_args_list.extend(extra_args)

  _inject_trailing_args(new_args_list, std_args_order, maps.target_inject_map, maps.target_val_map)

  for i in range(len(new_args_list) - 1):
    new_args_list[i] = new_args_list[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
  if new_args_list:
    new_args_list[-1] = new_args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

  return new_args_list
