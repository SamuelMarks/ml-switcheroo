"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments, convert_value_to_cst


def test_convert_value_to_cst_ast_literal_str():
  """Test element."""
  # If it parses to something else
  res = convert_value_to_cst("['a']")
  assert isinstance(res, cst.List)


def test_normalize_arguments_dict_variadic_default():
  """Test element."""
  original = cst.Call(func=cst.Name("foo"), args=[])
  config = {
    "signature": {
      "args": [{"name": "var_args", "is_variadic": True, "default": [1, 2]}],
      "pack_variadic_into_keyword": "packed_args",
      "pack_variadic_type": "List",
    },
    "library_to_std_args": {},
  }
  normalized = normalize_arguments(
    original, original, config, target_impl={}, source_fw="torch", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) == 0


def test_normalize_arguments_pack_variadic_one_element():
  """Test element."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a"))])
  config = {
    "signature": {
      "args": [{"name": "var_args", "is_variadic": True}],
      "pack_variadic_into_keyword": "packed_args",
      "pack_variadic_type": "Tuple",
    },
    "library_to_std_args": {},
  }
  normalize_arguments(original, original, config, target_impl={}, source_fw="torch", is_module_alias_fn=lambda x: False)
  pass


def test_normalize_arguments_pack_variadic_list():
  """Test element."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a"))])
  config = {
    "signature": {
      "args": [{"name": "var_args", "is_variadic": True}],
      "pack_variadic_into_keyword": "packed_args",
      "pack_variadic_type": "List",
    },
    "library_to_std_args": {},
  }
  normalize_arguments(original, original, config, target_impl={}, source_fw="torch", is_module_alias_fn=lambda x: False)
  pass


def test_normalize_arguments_method_call_module_alias():
  """Test element."""
  original = cst.Call(func=cst.Attribute(value=cst.Name("np"), attr=cst.Name("foo")), args=[cst.Arg(value=cst.Name("a"))])
  config = {"signature": {"args": ["self", "arg1"]}, "library_to_std_args": {}}

  normalized = normalize_arguments(
    original, original, config, target_impl={}, source_fw="numpy", is_module_alias_fn=lambda x: x == "np"
  )
  # np.foo(a) won't have self injected. So 2 arg in result actually, because receiver injection adds 'np'
  assert len(normalized) == 2


def test_normalize_arguments_method_call_no_std_args_order():
  """Test element."""
  original = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("foo")),
    args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))],
  )
  config = {"signature": {"args": []}, "library_to_std_args": {}}
  normalize_arguments(original, original, config, target_impl={}, source_fw="numpy", is_module_alias_fn=lambda x: False)


def test_normalize_arguments_extra_args_kwargs_map():
  """Test element."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))])
  config = {"signature": {"args": []}, "library_to_std_args": {}}
  target_impl = {"kwargs_map": {"kw": "new_kw"}}
  normalized = normalize_arguments(
    original, original, config, target_impl=target_impl, source_fw="numpy", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) == 1


def test_normalize_arguments_value_options_str_syntax_error():
  """Test element."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))])
  config = {"signature": {"args": ["kw"]}, "library_to_std_args": {}}
  target_impl = {"inject_args": {"kw": "invalid syntax!@"}}
  # It catches exception? actually inject args calls parse_expression! Let's just catch it in the test
  import pytest
  from libcst import ParserSyntaxError

  with pytest.raises(ParserSyntaxError):
    normalize_arguments(
      original, original, config, target_impl=target_impl, source_fw="numpy", is_module_alias_fn=lambda x: False
    )


def test_normalize_arguments_value_options_complex_obj():
  """Test element."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))])
  config = {"signature": {"args": ["kw"]}, "library_to_std_args": {}}
  target_impl = {"arg_values": {"kw": {"some": "dict", "complex": "value"}}}
  normalized = normalize_arguments(
    original, original, config, target_impl=target_impl, source_fw="numpy", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) == 1
