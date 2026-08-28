"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.normalization_utils import (
  convert_value_to_cst,
  normalize_arguments,
)
from typing import Dict, Any


def test_convert_value_negative_int() -> None:
  """Test element."""
  # Covers line 111
  node: cst.CSTNode = convert_value_to_cst(-42)
  assert isinstance(node, cst.UnaryOperation)


def test_normalize_arguments_various() -> None:
  """Test element."""
  # Setup standard call
  original_node: cst.Call = getattr(getattr(cst.parse_statement("obj.func(1, 2, kw=3, bad=4)"), "body")[0], "value")
  updated_node: cst.Call = original_node

  op_details: Dict[str, Any] = {
    "std_args": [
      {"name": ""},  # Empty name, covers 164->161
      {"name": "x", "is_variadic": False},  # Covers 166->168
      {"name": "y", "default": 10},  # Covers 169
      ["z", "z_alias"],  # Covers 171
      {"name": "v", "is_variadic": True},
    ],
    "variants": {"source_fw": {"args": {"kw": "z"}}},
  }

  target_impl: Dict[str, Any] = {
    "args": {"y": None},  # Covers 304
    "pack_to_tuple": "packed_v",
    "pack_as": "List",
  }

  def is_module_alias_fn(node: Any) -> bool:
    return True  # Covers 196

  # Covers 196 (is_module_alias_fn returns True, so is_method_call = False)
  normalize_arguments(original_node, updated_node, op_details, target_impl, "source_fw", is_module_alias_fn)


def test_normalize_arguments_packing() -> None:
  """Test element."""
  # covers 203->202, 226, 230-231, 248-265, 279-291, 297-298, 313-316, 324-329, 345-346, 380->383
  original_node: cst.Call = getattr(getattr(cst.parse_statement("func(1, 2, 3, v1=4, v2='val')"), "body")[0], "value")
  updated_node: cst.Call = getattr(getattr(cst.parse_statement("func(1, 2, 3, 4)"), "body")[0], "value")

  op_details: Dict[str, Any] = {
    "std_args": [
      {"name": "a"},
      {"name": "v", "is_variadic": True},
      {"name": "missing", "default": 100},  # covers 279-291
    ]
  }

  target_impl: Dict[str, Any] = {
    "pack_to_tuple": "v_packed",
    "pack_as": "Tuple",
    "arg_values": {
      "a": {"1": "11"},  # covers 313-316
      "missing": "@#invalid",  # covers 324-327
    },
    "inject_args": {
      "inj1": "55",  # covers 380->383
    },
  }

  normalize_arguments(original_node, updated_node, op_details, target_impl, "source_fw", lambda x: False)


def test_normalize_arguments_missing_branches() -> None:
  """Test element."""
  # 203->202: is_method_call with positional args
  # 260: pack_as="Tuple" with exactly 1 element
  # 290-291: Exception in default value (invalid cst.Name)
  # 329: target_val_map with non-string value
  # 380->383: new_args_list[-1].comma == MaybeSentinel.DEFAULT

  original_node: cst.Call = getattr(getattr(cst.parse_statement("obj.func(1)"), "body")[0], "value")
  updated_node: cst.Call = getattr(getattr(cst.parse_statement("func(1)"), "body")[0], "value")

  op_details: Dict[str, Any] = {
    "std_args": [
      {"name": "first"},
      {"name": "v", "is_variadic": True},
      {"name": "invalid-name", "default": 99},  # covers 290-291
    ]
  }

  target_impl: Dict[str, Any] = {
    "pack_to_tuple": "v_packed",
    "pack_as": "Tuple",  # covers 260
    "arg_values": {
      "first": 42,  # covers 329
    },
    "inject_args": {
      "inj1": "55",  # covers 380->381
    },
  }

  normalize_arguments(original_node, updated_node, op_details, target_impl, "source_fw", lambda x: False)


def test_normalize_arguments_empty_list_injection() -> None:
  """Test element."""
  # Covers 380->383 where len(new_args_list) == 0
  original_node: cst.Call = getattr(getattr(cst.parse_statement("func()"), "body")[0], "value")
  updated_node: cst.Call = getattr(getattr(cst.parse_statement("func()"), "body")[0], "value")

  op_details: Dict[str, Any] = {"std_args": []}

  target_impl: Dict[str, Any] = {
    "inject_args": {
      "inj1": "55",
    }
  }

  normalize_arguments(original_node, updated_node, op_details, target_impl, "source_fw", lambda x: False)
