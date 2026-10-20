"""Argument packing and unpacking for semantic resolution.

Maps source keyword arguments to abstract standard kwargs and vice versa,
handling variadic arguments and renaming.
"""

import typing

from pydantic import JsonValue

ArgVal: typing.TypeAlias = typing.Union[JsonValue, typing.Tuple["ArgVal", ...]]
MappingRule = typing.Union[str, typing.Mapping[str, str], JsonValue]


class ArgumentPacker:
  """Packs source kwargs into abstract standard kwargs."""

  def pack(
    self,
    source_kwargs: typing.Mapping[str, ArgVal],
    mapping_rules: typing.Mapping[str, MappingRule],
  ) -> typing.Dict[str, ArgVal]:
    """Pack source kwargs according to mapping rules.

    Args:
        source_kwargs: Keyword arguments from the source framework.
        mapping_rules: Rules defining how to map to abstract kwargs.

    Returns:
        Dictionary of packed abstract kwargs.
    """
    abstract_kwargs: typing.Dict[str, ArgVal] = {}
    packed_lists: typing.Dict[str, typing.List[ArgVal]] = {}
    for source_key, value in source_kwargs.items():
      if source_key in mapping_rules:
        target_rule = mapping_rules[source_key]
        if isinstance(target_rule, str):
          abstract_kwargs[target_rule] = value
        elif isinstance(target_rule, dict) and "pack_to" in target_rule:
          pack_key = str(target_rule["pack_to"])
          if pack_key not in packed_lists:
            packed_lists[pack_key] = []
          packed_lists[pack_key].append(value)

    # Convert packed lists to tuples if needed
    for key, val_list in packed_lists.items():
      abstract_kwargs[key] = tuple(val_list)

    return abstract_kwargs


class ArgumentUnpacker:
  """Unpacks abstract standard kwargs into target kwargs."""

  def unpack(
    self,
    abstract_kwargs: typing.Mapping[str, ArgVal],
    mapping_rules: typing.Mapping[str, MappingRule],
  ) -> typing.Dict[str, ArgVal]:
    """Unpack abstract kwargs according to mapping rules.

    Args:
        abstract_kwargs: Keyword arguments from the abstract operation.
        mapping_rules: Rules defining how to map from abstract kwargs.

    Returns:
        Dictionary of target kwargs.
    """
    target_kwargs: typing.Dict[str, ArgVal] = {}
    # Invert mapping rules for unpacking
    inverse_rules: typing.Dict[str, str] = {}
    for src_key, target_rule in mapping_rules.items():
      if isinstance(target_rule, str):
        inverse_rules[target_rule] = src_key
      elif isinstance(target_rule, dict) and "pack_to" in target_rule:
        pack_key = str(target_rule["pack_to"])
        inverse_rules[pack_key] = src_key

    for abs_key, value in abstract_kwargs.items():
      if abs_key in inverse_rules:
        target_key = inverse_rules[abs_key]
        # For variadic unpacked values passed as tuple, simply pass them through
        target_kwargs[target_key] = value

    return target_kwargs
