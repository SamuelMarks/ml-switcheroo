"""Module docstring."""

from typing import Any, Dict, List

import libcst as cst

from ml_switcheroo.core.rewriter.normalization_utils import (
  convert_value_to_cst,
  extract_primitive_key,
  normalize_arguments,
)


def test_extract_primitive_key_branches() -> None:
  """Docstring."""
  assert extract_primitive_key(cst.SimpleString('"test"')) == "test"
  assert extract_primitive_key(cst.Integer("1")) == "1"
  assert extract_primitive_key(cst.Name("x")) == "x"
  assert extract_primitive_key(cst.Float("1.0")) is None


def test_convert_value_to_cst_branches() -> None:
  """Docstring."""
  assert isinstance(convert_value_to_cst(True), cst.Name)
  assert isinstance(convert_value_to_cst(False), cst.Name)
  assert isinstance(convert_value_to_cst(None), cst.Name)
  assert isinstance(convert_value_to_cst(1), cst.Integer)
  assert isinstance(convert_value_to_cst(1.5), cst.Float)
  assert isinstance(convert_value_to_cst(-1.5), cst.UnaryOperation)
  assert isinstance(convert_value_to_cst("abc"), cst.SimpleString)

  lst_node: cst.CSTNode = convert_value_to_cst([1, 2])
  assert isinstance(lst_node, cst.List)
  assert len(lst_node.elements) == 2

  tup_node: cst.CSTNode = convert_value_to_cst((1,))
  assert isinstance(tup_node, cst.Tuple)

  dict_node: cst.CSTNode = convert_value_to_cst({"a": 1})
  assert isinstance(dict_node, cst.Dict)

  # Unknown
  class Unknown:
    """Docstring."""

    pass

  assert isinstance(convert_value_to_cst(Unknown()), cst.SimpleString)


# --- Merged from test_rewriter_normalization_utils_extra.py ---


def test_normalization_utils_extra() -> None:
  """Docstring."""
  code: str = "f(a=1)"
  original_node: cst.Call = getattr(cst.parse_statement(code).body[0], "value")
  updated_node: cst.Call = original_node

  op_details: Dict[str, Any] = {"std_args": ["a"]}
  target_impl: Dict[str, Any] = {"arg_values": {"a": "2", "extra": 10}}

  def is_module_alias_fn(x: Any) -> bool:
    """Function doc."""
    return False

  res: List[cst.Arg] = normalize_arguments(
    original_node, updated_node, op_details, target_impl, "torch", is_module_alias_fn
  )
  assert len(res) == 2


def test_convert_value_to_cst_empty_collections() -> None:
  """Function doc."""
  import libcst as cst

  from ml_switcheroo.core.rewriter.normalization_utils import convert_value_to_cst

  res_list: cst.CSTNode = convert_value_to_cst([])
  assert isinstance(res_list, cst.List) and len(res_list.elements) == 0

  res_tuple: cst.CSTNode = convert_value_to_cst(())
  assert isinstance(res_tuple, cst.Tuple) and len(res_tuple.elements) == 0

  res_dict: cst.CSTNode = convert_value_to_cst({})
  assert isinstance(res_dict, cst.Dict) and len(res_dict.elements) == 0


def test_normalize_args_implicit_receiver() -> None:
  """Function doc."""
  import libcst as cst

  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  # receiver_injected = True
  # This happens when implicit_receiver=True and the first standard arg isn't provided.
  # We need an Attribute call.
  node: cst.Call = getattr(getattr(cst.parse_statement("obj.func(a=1)"), "body")[0], "value")
  mapping: Dict[str, Any] = {"std_args": ["x", "a"], "lib_to_std": {"a": "a"}, "implicit_receiver": True}
  # obj should become x
  target_impl: Dict[str, Any] = {"std_args": ["x", "a"]}
  res: List[cst.Arg] = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args['x'] = obj
  assert len(res) == 2

  # Now let's test where it IS provided
  node2: cst.Call = getattr(getattr(cst.parse_statement("obj.func(x=2, a=1)"), "body")[0], "value")
  res2: List[cst.Arg] = normalize_arguments(node2, node2, mapping, target_impl, "torch", lambda x: False)
  # First std arg provided, so no implicit receiver injection
  assert len(res2) == 2

  # Now implicit_receiver = False and it is an Attribute
  mapping3: Dict[str, Any] = {"std_args": ["x"], "implicit_receiver": False}
  node3: cst.Call = getattr(getattr(cst.parse_statement("obj.func(x=1)"), "body")[0], "value")
  res3: List[cst.Arg] = normalize_arguments(node3, node3, mapping3, target_impl, "torch", lambda x: False)
  assert len(res3) == 1


def test_normalize_args_positional_duplicate() -> None:
  """Function doc."""
  import libcst as cst

  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  node: cst.Call = getattr(getattr(cst.parse_statement("func(1, 2)"), "body")[0], "value")
  mapping: Dict[str, Any] = {"std_args": ["x", "x"], "implicit_receiver": False}
  target_impl: Dict[str, Any] = {"std_args": ["x"]}
  res: List[cst.Arg] = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args will have 'x': 1, and the second 'x' will be ignored but pos_idx increments
  assert len(res) == 2


def test_normalize_args_empty_variadic() -> None:
  """Function doc."""
  import libcst as cst

  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  node: cst.Call = getattr(getattr(cst.parse_statement("func()"), "body")[0], "value")
  mapping: Dict[str, Any] = {"std_args": [{"name": "*args", "is_variadic": True}], "implicit_receiver": False}
  target_impl: Dict[str, Any] = {"std_args": [{"name": "args"}], "pack_to_tuple": "args", "pack_as": "Tuple"}
  res: List[cst.Arg] = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args will NOT have '*args' because packing_mode was not entered
  assert len(res) == 0


# --- Merged from test_rewriter_normalization_utils_extra3.py ---


def test_normalize_arguments_full() -> None:
  """Docstring."""
  original: cst.Call = cst.Call(
    func=cst.Name("foo"),
    args=[
      cst.Arg(value=cst.Name("val1")),
      cst.Arg(keyword=cst.Name("k1"), value=cst.Name("v1")),
      cst.Arg(keyword=cst.Name("k_extra"), value=cst.Name("v_extra")),
    ],
  )

  config: Dict[str, Any] = {
    "signature": {"args": [{"name": "arg1", "default": "def1"}, {"name": "arg2"}, ("arg3", "int"), "arg4"]},
    "library_to_std_args": {"k1": "arg2"},
    "target": {
      "arg_values": {"arg1": "new_def1", "arg2": {"v1": "target_v1"}},
      "kwargs_map": {"k_extra": None, "arg3": "new_arg3"},
      "inject_args": {"injected1": "inj_val"},
    },
  }

  normalized: List[cst.Arg] = normalize_arguments(
    original, original, config, target_impl=config["target"], source_fw="torch", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) > 0


# --- Merged from test_rewriter_normalization_utils_extra2.py ---


def test_value_to_cst_negative_float() -> None:
  """Docstring."""
  res: cst.CSTNode = convert_value_to_cst(-3.14)
  assert isinstance(res, cst.UnaryOperation)


def test_normalize_arguments_tuple_item() -> None:
  """Docstring."""
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
  """Docstring."""
  original: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("foo")), args=[cst.Arg(value=cst.Name("a"))]
  )
  config: Dict[str, Any] = {"signature": {"args": ["self", "arg1"]}, "library_to_std_args": {}}

  normalize_arguments(original, original, config, target_impl={}, source_fw="torch", is_module_alias_fn=lambda x: False)


def test_normalize_arguments_pack_variadic() -> None:
  """Docstring."""
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
  """Docstring."""
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


# --- Merged from test_rewriter_normalization_utils_extra5.py ---


def test_convert_value_negative_int() -> None:
  """Docstring."""
  # Covers line 111
  node: cst.CSTNode = convert_value_to_cst(-42)
  assert isinstance(node, cst.UnaryOperation)


def test_normalize_arguments_various() -> None:
  """Docstring."""
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
    """Docstring."""
    return True  # Covers 196

  # Covers 196 (is_module_alias_fn returns True, so is_method_call = False)
  normalize_arguments(original_node, updated_node, op_details, target_impl, "source_fw", is_module_alias_fn)


def test_normalize_arguments_packing() -> None:
  """Docstring."""
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
  """Docstring."""
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
  """Docstring."""
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


# --- Merged from test_rewriter_normalization_utils_extra4.py ---


def test_convert_value_to_cst_ast_literal_str() -> None:
  """Docstring."""
  # If it parses to something else
  res: cst.CSTNode = convert_value_to_cst("['a']")
  assert isinstance(res, cst.List)


def test_normalize_arguments_dict_variadic_default() -> None:
  """Docstring."""
  original: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  config: Dict[str, Any] = {
    "signature": {
      "args": [{"name": "var_args", "is_variadic": True, "default": [1, 2]}],
      "pack_variadic_into_keyword": "packed_args",
      "pack_variadic_type": "List",
    },
    "library_to_std_args": {},
  }
  normalized: List[cst.Arg] = normalize_arguments(
    original, original, config, target_impl={}, source_fw="torch", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) == 0


def test_normalize_arguments_pack_variadic_one_element() -> None:
  """Docstring."""
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
  pass


def test_normalize_arguments_pack_variadic_list() -> None:
  """Docstring."""
  original: cst.Call = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a"))])
  config: Dict[str, Any] = {
    "signature": {
      "args": [{"name": "var_args", "is_variadic": True}],
      "pack_variadic_into_keyword": "packed_args",
      "pack_variadic_type": "List",
    },
    "library_to_std_args": {},
  }
  normalize_arguments(original, original, config, target_impl={}, source_fw="torch", is_module_alias_fn=lambda x: False)
  pass


def test_normalize_arguments_method_call_module_alias() -> None:
  """Docstring."""
  original: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("np"), attr=cst.Name("foo")), args=[cst.Arg(value=cst.Name("a"))]
  )
  config: Dict[str, Any] = {"signature": {"args": ["self", "arg1"]}, "library_to_std_args": {}}

  normalized: List[cst.Arg] = normalize_arguments(
    original, original, config, target_impl={}, source_fw="numpy", is_module_alias_fn=lambda x: x == "np"
  )
  # np.foo(a) won't have self injected. So 2 arg in result actually, because receiver injection adds 'np'
  assert len(normalized) == 2


def test_normalize_arguments_method_call_no_std_args_order() -> None:
  """Docstring."""
  original: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("foo")),
    args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))],
  )
  config: Dict[str, Any] = {"signature": {"args": []}, "library_to_std_args": {}}
  normalize_arguments(original, original, config, target_impl={}, source_fw="numpy", is_module_alias_fn=lambda x: False)


def test_normalize_arguments_extra_args_kwargs_map() -> None:
  """Docstring."""
  original: cst.Call = cst.Call(func=cst.Name("foo"), args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))])
  config: Dict[str, Any] = {"signature": {"args": []}, "library_to_std_args": {}}
  target_impl: Dict[str, Any] = {"kwargs_map": {"kw": "new_kw"}}
  normalized: List[cst.Arg] = normalize_arguments(
    original, original, config, target_impl=target_impl, source_fw="numpy", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) == 1


def test_normalize_arguments_value_options_str_syntax_error() -> None:
  """Docstring."""
  original: cst.Call = cst.Call(func=cst.Name("foo"), args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))])
  config: Dict[str, Any] = {"signature": {"args": ["kw"]}, "library_to_std_args": {}}
  target_impl: Dict[str, Any] = {"inject_args": {"kw": "invalid syntax!@"}}
  # It catches exception? actually inject args calls parse_expression! Let's just catch it in the test
  import pytest
  from libcst import ParserSyntaxError

  with pytest.raises(ParserSyntaxError):
    normalize_arguments(
      original, original, config, target_impl=target_impl, source_fw="numpy", is_module_alias_fn=lambda x: False
    )


def test_normalize_arguments_value_options_complex_obj() -> None:
  """Docstring."""
  original: cst.Call = cst.Call(func=cst.Name("foo"), args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))])
  config: Dict[str, Any] = {"signature": {"args": ["kw"]}, "library_to_std_args": {}}
  target_impl: Dict[str, Any] = {"arg_values": {"kw": {"some": "dict", "complex": "value"}}}
  normalized: List[cst.Arg] = normalize_arguments(
    original, original, config, target_impl=target_impl, source_fw="numpy", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) == 1


def test_convert_value_to_cst_literal_eval_same_str() -> None:
  """Test convert_value_to_cst with a str subclass where ast.literal_eval returns an equal string."""

  class _EqualStr(str):
    """String subclass returning False for inequality to test identical string branch."""

    def __ne__(self, other: Any) -> bool:
      """Check inequality."""
      return False

  val = _EqualStr("'test'")
  res = convert_value_to_cst(val)
  assert isinstance(res, cst.SimpleString)


def test_normalize_arguments_dynamic_receiver_attribute_check() -> None:
  """Test normalize_arguments when original_node.func ceases to be an attribute mid-processing."""

  class _DynamicFuncCall:
    """Mock call with dynamic func property."""

    def __init__(self, first_func: cst.CSTNode, second_func: cst.CSTNode, args: List[cst.Arg]) -> None:
      """Initialize mock call."""
      self._first_func = first_func
      self._second_func = second_func
      self.args = args
      self._count = 0

    @property
    def func(self) -> cst.CSTNode:
      """Return first_func on initial checks, then second_func."""
      self._count += 1
      if self._count <= 2:
        return self._first_func
      return self._second_func

  # Branch 211->220: std_args_order present, but func becomes non-attribute
  mock_call = _DynamicFuncCall(
    cst.Attribute(value=cst.Name("obj"), attr=cst.Name("method")),
    cst.Name("not_attr"),
    [cst.Arg(value=cst.Name("val"))],
  )
  config: Dict[str, Any] = {"std_args": ["first", "second"], "library_to_std_args": {}}
  res1 = normalize_arguments(
    mock_call,  # type: ignore[arg-type]
    cst.Call(func=cst.Name("method"), args=[cst.Arg(value=cst.Name("val"))]),
    config,
    target_impl={},
    source_fw="torch",
    is_module_alias_fn=lambda _: False,
  )
  assert len(res1) == 1

  # Branch 216->220: std_args_order empty, but func becomes non-attribute
  mock_call_no_std = _DynamicFuncCall(
    cst.Attribute(value=cst.Name("obj"), attr=cst.Name("method")),
    cst.Name("not_attr"),
    [cst.Arg(value=cst.Name("val"))],
  )
  config_no_std: Dict[str, Any] = {"std_args": [], "library_to_std_args": {}}
  res2 = normalize_arguments(
    mock_call_no_std,  # type: ignore[arg-type]
    cst.Call(func=cst.Name("method"), args=[cst.Arg(value=cst.Name("val"))]),
    config_no_std,
    target_impl={},
    source_fw="torch",
    is_module_alias_fn=lambda _: False,
  )
  assert len(res2) == 1


def test_normalize_arguments_val_options_dict_unmatched_key() -> None:
  """Test val_options dictionary where primitive key is not present in dictionary."""
  call = cst.Call(func=cst.Name("foo"), args=[cst.Arg(keyword=cst.Name("mode"), value=cst.SimpleString("'unmatched'"))])
  config: Dict[str, Any] = {"std_args": ["mode"], "library_to_std_args": {}}
  target_impl: Dict[str, Any] = {"arg_values": {"mode": {"known": "1"}}}
  res = normalize_arguments(
    call,
    call,
    config,
    target_impl=target_impl,
    source_fw="torch",
    is_module_alias_fn=lambda _: False,
  )
  assert len(res) == 1
