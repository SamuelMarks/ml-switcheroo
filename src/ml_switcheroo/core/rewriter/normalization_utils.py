"""
Normalization Utilities.

This module provides helper functions for converting between Python runtime values
and LibCST Abstract Syntax Tree nodes. It supports primitive types (int, float,
bool, str) and recursive container types (list, tuple, dict).
"""

import json
from typing import Any, Optional

import libcst as cst


def extract_primitive_key(node: cst.BaseExpression) -> Optional[str]:
  """
  Extracts a string representation of a primitive AST node for Enum key lookup.

  Handles simple strings strings, integers, and simple names (identifiers).

  Args:
      node: The CST expression node.

  Returns:
      The string value (if literal) or variable name. Returns None
      if the node type is complex or unsupported.
  """
  if isinstance(node, cst.SimpleString):
    return node.value.strip("'").strip('"')
  elif isinstance(node, cst.Integer):
    return node.value
  elif isinstance(node, cst.Name):
    return node.value
  return None


def convert_value_to_cst(val: Any) -> cst.BaseExpression:
  """
  Recursively converts a python value (primitive/container) to a CST literal expression node.

  Supported types:
  - Primitives: bool, int, float, str, None
  - Containers: list, tuple, dict

  Args:
      val: The python value to convert.

  Returns:
      The corresponding LibCST node.
  """
  # 1. Container Recursion (List/Tuple)
  if isinstance(val, (list, tuple)):
    elements = []
    for item in val:
      node = convert_value_to_cst(item)
      # Add comma with spacing
      elements.append(
        cst.Element(
          value=node,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    # Fix trailing comma for the last element based on Python style preferences
    # Usually strict JSON-like structures don't strictly need one, but LibCST allows exact control.
    if elements:
      last = elements[-1]
      elements[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    if isinstance(val, list):
      return cst.List(elements=elements)
    else:
      return cst.Tuple(elements=elements)

  # 2. Key-Value Recursion (Dict)
  if isinstance(val, dict):
    elements = []
    for k, v in val.items():
      k_node = convert_value_to_cst(k)
      v_node = convert_value_to_cst(v)
      elements.append(
        cst.DictElement(
          key=k_node,
          value=v_node,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    if elements:
      last = elements[-1]
      elements[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return cst.Dict(elements=elements)

  # 3. Primitives
  # Important: bool MUST be checked before int because bool is subclass of int in Python
  if isinstance(val, bool):
    return cst.Name("True") if val else cst.Name("False")
  elif isinstance(val, int):
    return cst.Integer(str(val))
  elif isinstance(val, float):
    return cst.Float(repr(val))
  elif isinstance(val, str):
    # Use json.dumps to ensure proper quoting and escaping (produces double quotes)
    return cst.SimpleString(json.dumps(val))
  elif val is None:
    return cst.Name("None")
  else:
    # Fallback for unknown objects using string representation
    return cst.SimpleString(repr(str(val)))
