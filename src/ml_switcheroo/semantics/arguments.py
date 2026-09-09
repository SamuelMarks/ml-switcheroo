"""Argument packing and unpacking for semantic resolution.

Maps source keyword arguments to abstract standard kwargs and vice versa,
handling variadic arguments and renaming.
"""

from typing import Dict, Any


class ArgumentPacker:
  """Packs source kwargs into abstract standard kwargs."""

  def pack(self, source_kwargs: Dict[str, Any], mapping_rules: Dict[str, Any]) -> Dict[str, Any]:
    """Pack source kwargs according to mapping rules.

    Args:
        source_kwargs: Keyword arguments from the source framework.
        mapping_rules: Rules defining how to map to abstract kwargs.

    Returns:
        Dictionary of packed abstract kwargs.
    """
    abstract_kwargs: Dict[str, Any] = {}
    for source_key, value in source_kwargs.items():
      if source_key in mapping_rules:
        target_rule = mapping_rules[source_key]
        if isinstance(target_rule, str):
          abstract_kwargs[target_rule] = value
        elif isinstance(target_rule, dict) and "pack_to" in target_rule:
          pack_key = str(target_rule["pack_to"])
          if pack_key not in abstract_kwargs:
            abstract_kwargs[pack_key] = []
          abstract_kwargs[pack_key].append(value)

    # Convert packed lists to tuples if needed
    for key, val in abstract_kwargs.items():
      if isinstance(val, list):
        abstract_kwargs[key] = tuple(val)

    return abstract_kwargs


class ArgumentUnpacker:
  """Unpacks abstract standard kwargs into target kwargs."""

  def unpack(self, abstract_kwargs: Dict[str, Any], mapping_rules: Dict[str, Any]) -> Dict[str, Any]:
    """Unpack abstract kwargs according to mapping rules.

    Args:
        abstract_kwargs: Keyword arguments from the abstract operation.
        mapping_rules: Rules defining how to map from abstract kwargs.

    Returns:
        Dictionary of target kwargs.
    """
    target_kwargs: Dict[str, Any] = {}
    # Invert mapping rules for unpacking
    inverse_rules: Dict[str, Any] = {}
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
