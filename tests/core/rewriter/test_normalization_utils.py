"""Test suite for normalization_utils.py"""

import libcst as cst

from ml_switcheroo.core.rewriter.normalization_utils import (
  extract_primitive_key,
  convert_value_to_cst,
  normalize_arguments,
)


def parse_call(code: str) -> cst.Call:
  """Docstring."""
  module = cst.parse_module(code)
  return module.body[0].body[0].value


def test_extract_primitive_key():
  """Docstring."""
  assert extract_primitive_key(cst.SimpleString('"hi"')) == "hi"
  assert extract_primitive_key(cst.SimpleString("'hi'")) == "hi"
  assert extract_primitive_key(cst.Integer("42")) == "42"
  assert extract_primitive_key(cst.Name("x")) == "x"
  assert extract_primitive_key(cst.List([])) is None


def test_convert_value_to_cst():
  """Docstring."""
  # Primitives
  assert convert_value_to_cst(True).value == "True"
  assert convert_value_to_cst(False).value == "False"
  assert convert_value_to_cst(42).value == "42"
  assert convert_value_to_cst(3.14).value == "3.14"
  assert convert_value_to_cst("hello").value == '"hello"'
  assert convert_value_to_cst(None).value == "None"

  # Fallback
  fallback = convert_value_to_cst(object())
  assert isinstance(fallback, cst.SimpleString)

  # Lists/Tuples
  lst = convert_value_to_cst([1, 2])
  assert isinstance(lst, cst.List)
  assert len(lst.elements) == 2
  assert lst.elements[0].value.value == "1"

  tpl = convert_value_to_cst((1,))
  assert isinstance(tpl, cst.Tuple)
  assert len(tpl.elements) == 1

  # Dict
  d = convert_value_to_cst({"a": 1})
  assert isinstance(d, cst.Dict)
  assert len(d.elements) == 1
  assert d.elements[0].key.value == '"a"'
  assert d.elements[0].value.value == "1"


def test_normalize_arguments_basic():
  """Docstring."""
  original = parse_call("add(x, y)")
  updated = parse_call("add(x, y)")
  details = {"std_args": ["a", "b"]}
  target_impl = {"args": {"a": "left", "b": "right"}}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 2
  assert result[0].value.value == "x"
  assert result[1].value.value == "y"


def test_normalize_arguments_receiver():
  """Docstring."""
  original = parse_call("x.add(y)")
  updated = parse_call("x.add(y)")
  details = {"std_args": ["a", "b"]}
  target_impl = {"args": {"a": "left", "b": "right"}}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 2
  assert result[0].value.value == "x"
  assert result[1].value.value == "y"


def test_normalize_arguments_receiver_no_std_args():
  """Docstring."""
  original = parse_call("x.add(y)")
  updated = parse_call("x.add(y)")
  details = {}
  target_impl = {}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 2  # y from args, x from receiver (appended to extra_args)


def test_normalize_arguments_kwargs():
  """Docstring."""
  original = parse_call("func(a=1, b=2)")
  updated = parse_call("func(a=1, b=2)")
  details = {"std_args": ["a", "b"], "variants": {"torch": {"args": {"a": "a", "b": "b"}}}}
  target_impl = {"args": {"a": "x", "b": "y"}}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 2
  assert result[0].keyword.value == "x"
  assert result[1].keyword.value == "y"


def test_normalize_arguments_packing():
  """Docstring."""
  original = parse_call("func(1, 2, 3)")
  updated = parse_call("func(1, 2, 3)")
  details = {"std_args": [{"name": "a", "is_variadic": True}]}
  target_impl = {"args": {"a": "a"}, "pack_to_tuple": "axes", "pack_as": "Tuple"}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 1
  assert result[0].keyword.value == "axes"
  assert isinstance(result[0].value, cst.Tuple)
  assert len(result[0].value.elements) == 3


def test_normalize_arguments_packing_list():
  """Docstring."""
  original = parse_call("func(1)")
  updated = parse_call("func(1)")
  details = {"std_args": [{"name": "a", "is_variadic": True}]}
  target_impl = {"args": {"a": "a"}, "pack_to_tuple": "axes", "pack_as": "List"}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.List)


def test_normalize_arguments_defaults():
  """Docstring."""
  original = parse_call("func()")
  updated = parse_call("func()")
  details = {"std_args": [{"name": "a", "default": 5}]}
  target_impl = {"args": {"a": "x"}}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 1
  assert result[0].keyword.value == "x"
  assert result[0].value.value == "5"


def test_normalize_arguments_val_map():
  """Docstring."""
  original = parse_call("func(a='fast')")
  updated = parse_call("func(a='fast')")
  details = {"std_args": ["a"]}
  target_impl = {"args": {"a": "mode"}, "arg_values": {"a": {"fast": "1", "slow": "0"}}}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 1
  assert result[0].keyword.value == "mode"
  assert result[0].value.value == "1"


def test_normalize_arguments_val_override():
  """Docstring."""
  original = parse_call("func(a=1)")
  updated = parse_call("func(a=1)")
  details = {"std_args": ["a"]}
  target_impl = {"args": {"a": "a"}, "arg_values": {"a": "True"}}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 1
  assert result[0].value.value == "True"


def test_normalize_arguments_inject_args():
  """Docstring."""
  original = parse_call("func()")
  updated = parse_call("func()")
  details = {"std_args": []}
  target_impl = {"inject_args": {"injected": "True"}}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 1
  assert result[0].keyword.value == "injected"
  assert result[0].value.value == "True"


def test_normalize_arguments_kwargs_map():
  """Docstring."""
  original = parse_call("func(a=1, drop_me=2)")
  updated = parse_call("func(a=1, drop_me=2)")
  details = {"std_args": ["a"]}
  target_impl = {"args": {"a": "a"}, "kwargs_map": {"drop_me": None}}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 1
  assert result[0].keyword.value == "a"


def test_normalize_arguments_ignore_alias():
  """Docstring."""
  original = parse_call("np.func(a=1)")
  updated = parse_call("np.func(a=1)")
  details = {"std_args": ["a"]}
  target_impl = {"args": {"a": "a"}}

  # is_module_alias returns True for "np"
  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: x.value == "np")
  assert len(result) == 1
  assert result[0].keyword.value == "a"


def test_normalize_arguments_none_alias():
  """Docstring."""
  original = parse_call("func(a=1)")
  updated = parse_call("func(a=1)")
  details = {"std_args": ["a"]}
  target_impl = {"args": {"a": None}}  # Alias None skips it

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 0


def test_normalize_arguments_val_map_str_fallback():
  """Docstring."""
  original = parse_call("func(a=1)")
  updated = parse_call("func(a=1)")
  details = {"std_args": ["a"]}
  target_impl = {"args": {"a": "a"}, "arg_values": {"a": "invalid syntax!"}}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.SimpleString)


def test_normalize_arguments_val_map_not_str():
  """Docstring."""
  original = parse_call("func(a=1)")
  updated = parse_call("func(a=1)")
  details = {"std_args": ["a"]}
  target_impl = {"args": {"a": "a"}, "arg_values": {"a": 99}}

  result = normalize_arguments(original, updated, details, target_impl, "torch", lambda x: False)
  assert len(result) == 1
  assert result[0].value.value == "99"


def test_normalization_utils_extra():
  """Docstring."""
  # 156: empty name dict
  details = {"std_args": [{"name": ""}]}
  normalize_arguments(parse_call("f()"), parse_call("f()"), details, {}, "torch", lambda x: False)

  # 189-193:
  details = {"std_args": ["a"]}
  target = {"args": {"a": "a"}}
  normalize_arguments(parse_call("obj.f(b=1)"), parse_call("obj.f(b=1)"), details, target, "torch", lambda x: False)

  # 246:
  details = {"std_args": [{"name": "a", "is_variadic": True}]}
  target = {"args": {"a": "a"}, "pack_to_tuple": "axes", "pack_as": "List"}
  normalize_arguments(parse_call("f()"), parse_call("f()"), details, target, "torch", lambda x: False)

  # 276:
  class BadCST:
    def __repr__(self):
      raise Exception("fail")

  details = {"std_args": [{"name": "a", "default": BadCST()}]}
  normalize_arguments(parse_call("f()"), parse_call("f()"), details, {"args": {"a": "a"}}, "torch", lambda x: False)

  # 331:
  details = {"std_args": ["a"]}
  target = {"args": {"a": "a"}}
  # value is unchanged
  original = parse_call("f(a=1)")
  normalize_arguments(original, original, details, target, "torch", lambda x: False)

  # 344:
  details = {"std_args": ["a"]}
  target = {"args": {"a": "a"}, "kwargs_map": {"drop": None, "keep": "keep"}}
  normalize_arguments(
    parse_call("f(a=1, drop=2, keep=3)"), parse_call("f(a=1, drop=2, keep=3)"), details, target, "torch", lambda x: False
  )

  # 356, 364:
  details = {"std_args": ["a"]}
  target = {"args": {"a": "a"}, "inject_args": {"b": "True"}}
  normalize_arguments(parse_call("f(a=1)"), parse_call("f(a=1)"), details, target, "torch", lambda x: False)


def test_normalize_arguments_pack_to_tuple_empty():
  """Docstring."""
  details = {"std_args": [{"name": "a", "is_variadic": True}]}
  target = {"args": {"a": "a"}, "pack_to_tuple": "axes", "pack_as": "List"}
  normalize_arguments(parse_call("f()"), parse_call("f()"), details, target, "torch", lambda x: False)


def test_normalization_utils_more_coverage():
  """Docstring."""
  # 156: empty string in dict name
  details = {"std_args": [{"name": ""}]}
  normalize_arguments(parse_call("f()"), parse_call("f()"), details, {}, "torch", lambda x: False)

  # 192:
  # if isinstance(original_node.func, cst.Attribute): ...
  # Wait, my previous test tested it but missed line 192.
  # The condition is: if not arg_provided: if isinstance(...): rec = ... found_args...
  original = parse_call("x.add()")
  updated = parse_call("x.add()")
  details = {"std_args": ["a"]}
  target = {"args": {"a": "left"}}
  normalize_arguments(original, updated, details, target, "torch", lambda x: False)

  # 246: is_list = pack_as_type == "List"
  original = parse_call("func(1, 2)")
  details = {"std_args": [{"name": "a", "is_variadic": True}]}
  target = {"args": {"a": "a"}, "pack_to_tuple": "axes", "pack_as": "List"}
  normalize_arguments(original, original, details, target, "torch", lambda x: False)

  # 331:
  details = {"std_args": ["a"]}
  target = {"args": {"a": "a"}}
  original = parse_call("func(1)")  # not a keyword arg!
  normalize_arguments(original, original, details, target, "torch", lambda x: False)

  # 356:
  details = {"std_args": []}
  target = {"inject_args": {"b": "cst.Name('True')"}}
  normalize_arguments(parse_call("func()"), parse_call("func()"), details, target, "torch", lambda x: False)

  # 364:
  details = {"std_args": []}
  target = {"inject_args": {"b": "1", "c": "2"}}
  original = parse_call("func(a=1,)")
  normalize_arguments(original, original, details, target, "torch", lambda x: False)
