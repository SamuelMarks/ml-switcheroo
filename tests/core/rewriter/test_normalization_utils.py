"""Test suite for the Normalization Utils module."""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter.normalization_utils import (
  extract_primitive_key,
  convert_value_to_cst,
  normalize_arguments,
)


def test_extract_primitive_key():
  """Extracts primitive key."""
  assert extract_primitive_key(cst.SimpleString('"foo"')) == "foo"
  assert extract_primitive_key(cst.SimpleString("'bar'")) == "bar"
  assert extract_primitive_key(cst.Integer("42")) == "42"
  assert extract_primitive_key(cst.Name("baz")) == "baz"
  assert extract_primitive_key(cst.Float("1.5")) is None


def test_convert_value_to_cst_primitives():
  """Converts value to cst primitives."""
  assert isinstance(convert_value_to_cst(True), cst.Name)
  assert convert_value_to_cst(True).value == "True"
  assert isinstance(convert_value_to_cst(False), cst.Name)
  assert convert_value_to_cst(False).value == "False"
  assert isinstance(convert_value_to_cst(42), cst.Integer)
  assert convert_value_to_cst(42).value == "42"
  assert isinstance(convert_value_to_cst(1.5), cst.Float)
  assert convert_value_to_cst(1.5).value == "1.5"
  assert isinstance(convert_value_to_cst("foo"), cst.SimpleString)
  assert convert_value_to_cst("foo").value == '"foo"'
  assert isinstance(convert_value_to_cst(None), cst.Name)
  assert convert_value_to_cst(None).value == "None"


def test_convert_value_to_cst_containers():
  """Converts value to cst containers."""
  lst = convert_value_to_cst([1, 2])
  assert isinstance(lst, cst.List)
  assert len(lst.elements) == 2
  assert getattr(lst.elements[-1].comma, "whitespace_after", None) is None
  t = convert_value_to_cst((1, 2))
  assert isinstance(t, cst.Tuple)
  assert len(t.elements) == 2
  d = convert_value_to_cst({"a": 1})
  assert isinstance(d, cst.Dict)
  assert len(d.elements) == 1
  assert d.elements[0].key.value == '"a"'


def test_convert_value_to_cst_fallback():
  """Converts value to cst fallback."""

  class Dummy:
    """Dummy."""

    def __str__(self):
      """Str."""
      return "dummy"

  node = convert_value_to_cst(Dummy())
  assert isinstance(node, cst.SimpleString)
  assert "dummy" in node.value


def test_normalize_arguments_basic():
  """Verifies the behavior of normalize arguments basic."""
  code = "foo(1, 2)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": ["a", "b"], "variants": {"torch": {"args": {"a": "a", "b": "b"}}}}
  target_impl = {"args": {"a": "ta", "b": "tb"}}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert len(new_args) == 2
  assert new_args[0].value.value == "1"
  assert new_args[1].value.value == "2"


def test_normalize_arguments_method_injection():
  """Verifies the behavior of normalize arguments method injection."""
  code = "x.add(2)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": ["a", "b"]}
  new_args = normalize_arguments(call_node, call_node, op_details, {}, "torch", lambda x: False)
  assert len(new_args) == 2
  assert getattr(new_args[0].value, "value", None) == "x"
  assert getattr(new_args[1].value, "value", None) == "2"


def test_normalize_arguments_packing():
  """Verifies the behavior of normalize arguments packing."""
  code = "foo(1, 2, 3)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": [{"name": "a", "is_variadic": True}]}
  target_impl = {"pack_to_tuple": "dims", "pack_as": "List"}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert len(new_args) == 1
  assert new_args[0].keyword.value == "dims"
  assert isinstance(new_args[0].value, cst.List)
  assert len(new_args[0].value.elements) == 3


def test_normalize_arguments_packing_tuple_single():
  """Verifies the behavior of normalize arguments packing tuple single."""
  code = "foo(1)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": [{"name": "a", "is_variadic": True}]}
  target_impl = {"pack_to_tuple": "dims", "pack_as": "Tuple"}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert isinstance(new_args[0].value, cst.Tuple)
  assert isinstance(new_args[0].value.elements[-1].comma, cst.Comma)


def test_normalize_arguments_defaults():
  """Verifies the behavior of normalize arguments defaults."""
  code = "foo()"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": [{"name": "a", "default": 42}]}
  new_args = normalize_arguments(call_node, call_node, op_details, {}, "torch", lambda x: False)
  assert len(new_args) == 1
  assert new_args[0].keyword.value == "a"
  assert new_args[0].value.value == "42"


def test_normalize_arguments_target_val_map():
  """Verifies the behavior of normalize arguments target value map."""
  code = "foo(x=1)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": ["x"]}
  target_impl = {"arg_values": {"x": {"1": "custom(1)"}}}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert new_args[0].keyword.value == "x"
  assert isinstance(new_args[0].value, cst.Call)
  assert new_args[0].value.func.value == "custom"


def test_normalize_arguments_target_val_map_literal():
  """Verifies the behavior of normalize arguments target value map literal."""
  code = "foo(x=1)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": ["x"]}
  target_impl = {"arg_values": {"x": "False"}}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert new_args[0].value.value == "False"


def test_normalize_arguments_target_val_map_literal_invalid_expr():
  """Verifies the behavior of normalize arguments target value map literal invalid expr."""
  code = "foo(x=1)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": ["x"]}
  target_impl = {"arg_values": {"x": "some invalid code ["}}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert isinstance(new_args[0].value, cst.SimpleString)


def test_normalize_arguments_target_val_map_cst():
  """Verifies the behavior of normalize arguments target value map cst."""
  code = "foo(x=1)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": ["x"]}
  target_impl = {"arg_values": {"x": 42}}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert new_args[0].value.value == "42"


def test_normalize_arguments_kwargs_filter():
  """Verifies the behavior of normalize arguments keyword arguments filter."""
  code = "foo(extra=1)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": []}
  target_impl = {"kwargs_map": {"extra": None}}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert len(new_args) == 0


def test_normalize_arguments_inject_args():
  """Verifies the behavior of normalize arguments inject arguments."""
  code = "foo()"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": []}
  target_impl = {"inject_args": {"new_arg": "100"}}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert len(new_args) == 1
  assert new_args[0].keyword.value == "new_arg"
  assert new_args[0].value.value == "100"


def test_normalize_arguments_target_val_map_inject():
  """Verifies the behavior of normalize arguments target value map inject."""
  code = "foo()"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": []}
  target_impl = {"arg_values": {"new_arg2": 42}}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert len(new_args) == 1
  assert new_args[0].keyword.value == "new_arg2"
  assert new_args[0].value.value == "42"


def test_normalize_arguments_drop_alias():
  """Verifies the behavior of normalize arguments drop alias."""
  code = "foo(a=1)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": ["a"]}
  target_impl = {"args": {"a": None}}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert len(new_args) == 0


def test_normalize_arguments_positional_change():
  """Verifies the behavior of normalize arguments positional change."""
  code = "foo(1)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": ["a"]}
  target_impl = {"arg_values": {"a": 42}}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert new_args[0].keyword is None
  assert new_args[0].value.value == "42"


def test_normalize_arguments_method_alias():
  """Verifies the behavior of normalize arguments method alias."""
  code = "torch.add(1, 2)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": ["a", "b"]}
  new_args = normalize_arguments(call_node, call_node, op_details, {}, "torch", lambda x: True)
  assert len(new_args) == 2
  assert new_args[0].value.value == "1"


def test_normalize_arguments_packing_empty():
  """Verifies the behavior of normalize arguments packing empty."""
  code = "foo()"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": [{"name": "a", "is_variadic": True}]}
  target_impl = {"pack_to_tuple": "dims", "pack_as": "List"}
  new_args = normalize_arguments(call_node, call_node, op_details, target_impl, "torch", lambda x: False)
  assert len(new_args) == 0


def test_normalize_arguments_extra_args():
  """Verifies the behavior of normalize arguments extra arguments."""
  code = "foo(1, extra=2)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value
  op_details = {"std_args": ["a"]}
  new_args = normalize_arguments(call_node, call_node, op_details, {}, "torch", lambda x: False)
  assert len(new_args) == 2
  assert new_args[1].keyword.value == "extra"


def test_normalize_arguments_default_exception():
  """Verifies the behavior of normalize arguments default correctly handling an exception."""
  code = "foo()"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value

  class Unconvertible:
    """Unconvertible."""

    pass

  op_details = {"std_args": [{"name": "a", "default": Unconvertible()}]}
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.normalization_utils as utils

    m.setattr(utils, "convert_value_to_cst", lambda x: 1 / 0)
    new_args = normalize_arguments(call_node, call_node, op_details, {}, "torch", lambda x: False)
  assert len(new_args) == 0
