"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments, convert_value_to_cst
from typing import List, Dict, Any


def test_value_to_cst_negative_float() -> None:
  """Test element."""
  res: cst.CSTNode = convert_value_to_cst(-3.14)
  assert isinstance(res, cst.UnaryOperation)


def test_normalize_arguments_tuple_item() -> None:
  """Test element."""
  original: cst.Call = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a"))])
  config: Dict[str, Any] = {
    "signature": {"args": [("my_arg", "int"), {"name": "my_opt_arg", "default": 42}]},
    "library_to_std_args": {},
  }
  normalized: List[cst.Arg] = normalize_arguments(
    original, original, config, target_impl={}, source_fw="torch", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) == 1


def test_normalize_arguments_method_call() -> None:
  """Test element."""
  original: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("foo")), args=[cst.Arg(value=cst.Name("a"))]
  )
  config: Dict[str, Any] = {"signature": {"args": ["self", "arg1"]}, "library_to_std_args": {}}

  normalize_arguments(original, original, config, target_impl={}, source_fw="torch", is_module_alias_fn=lambda x: False)


def test_normalize_arguments_pack_variadic() -> None:
  """Test element."""
  original: cst.Call = cst.Call(
    func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a")), cst.Arg(value=cst.Name("b")), cst.Arg(value=cst.Name("c"))]
  )
  config: Dict[str, Any] = {
    "signature": {
      "args": [{"name": "var_args", "is_variadic": True}],
      "pack_variadic_into_keyword": "packed_args",
      "pack_variadic_type": "List",
    },
    "library_to_std_args": {},
  }
  normalize_arguments(original, original, config, target_impl={}, source_fw="torch", is_module_alias_fn=lambda x: False)


def test_normalize_arguments_pack_variadic_tuple() -> None:
  """Test element."""
  original: cst.Call = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a"))])
  config: Dict[str, Any] = {
    "signature": {
      "args": [{"name": "var_args", "is_variadic": True}],
      "pack_variadic_into_keyword": "packed_args",
      "pack_variadic_type": "Tuple",
    },
    "library_to_std_args": {},
  }
  normalize_arguments(original, original, config, target_impl={}, source_fw="torch", is_module_alias_fn=lambda x: False)
