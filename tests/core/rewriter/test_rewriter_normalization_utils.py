"""Test suite for rewriter normalization utils."""

import libcst as _cst

from ml_switcheroo.core.rewriter.normalization_utils import (
  convert_value_to_cst,
  extract_primitive_key,
  normalize_arguments,
)
from ml_switcheroo.enums import RuntimeTensorOrValue

cst = _cst


def test_extract_primitive_key_branches() -> None:
  """Verify `extract_primitive_key` handles branches as expected."""
  assert extract_primitive_key(cst.SimpleString('"test"')) == "test"
  assert extract_primitive_key(cst.Integer("1")) == "1"
  assert extract_primitive_key(cst.Name("x")) == "x"
  assert extract_primitive_key(cst.Float("1.0")) is None


def test_convert_value_to_cst_branches() -> None:
  """Verify `convert_value_to_cst` returns `2` for branches."""
  assert isinstance(convert_value_to_cst(True), cst.Name)
  assert isinstance(convert_value_to_cst(False), cst.Name)
  assert isinstance(convert_value_to_cst(None), cst.Name)
  assert isinstance(convert_value_to_cst(1), cst.Integer)
  assert isinstance(convert_value_to_cst(1.5), cst.Float)
  assert isinstance(convert_value_to_cst(-1.5), cst.UnaryOperation)
  assert isinstance(convert_value_to_cst("abc"), cst.SimpleString)

  lst_node = convert_value_to_cst([1, 2])
  assert isinstance(lst_node, cst.List)
  assert len(lst_node.elements) == 2

  tup_node = convert_value_to_cst((1,))
  assert isinstance(tup_node, cst.Tuple)

  dict_node = convert_value_to_cst({"a": 1})
  assert isinstance(dict_node, cst.Dict)

  # Unknown
  class Unknown:
    """Sentinel `Unknown` fixture class used in rewriter normalization utils tests."""

  assert isinstance(convert_value_to_cst(Unknown()), cst.SimpleString)


def test_normalization_utils_extra() -> None:
  """Verify `normalize_arguments` returns `2` for normalization utils extra."""
  code = "f(a=1)"
  original_node = cst.parse_expression(code)
  assert isinstance(original_node, cst.Call)
  updated_node = original_node

  op_details = {"std_args": ["a"]}
  target_impl = {"arg_values": {"a": "2", "extra": 10}}

  def is_module_alias_fn(x: cst.CSTNode) -> bool:
    """Execute `is_module_alias_fn` helper in rewriter normalization utils tests.

    Args:
        x: Input tensor or scalar operand (`cst.CSTNode`).

    Returns:
        Resulting `bool` from is module alias fn.
    """
    return False

  res = normalize_arguments(original_node, updated_node, op_details, target_impl, "torch", is_module_alias_fn)
  assert len(res) == 2


def test_convert_value_to_cst_empty_collections() -> None:
  """Verify `convert_value_to_cst` returns `0` for empty collections."""
  import libcst as cst

  from ml_switcheroo.core.rewriter.normalization_utils import convert_value_to_cst

  res_list = convert_value_to_cst([])
  assert isinstance(res_list, cst.List)
  assert len(res_list.elements) == 0

  res_tuple = convert_value_to_cst(())
  assert isinstance(res_tuple, cst.Tuple)
  assert len(res_tuple.elements) == 0

  res_dict = convert_value_to_cst({})
  assert isinstance(res_dict, cst.Dict)
  assert len(res_dict.elements) == 0


def test_normalize_args_implicit_receiver() -> None:
  """Verify `normalize_arguments` returns `2` for normalize args implicit receiver."""
  import libcst as cst

  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  # receiver_injected = True
  # This happens when implicit_receiver=True and the first standard arg isn't provided.
  # We need an Attribute call.
  node = cst.parse_expression("obj.func(a=1)")
  assert isinstance(node, cst.Call)
  mapping = {"std_args": ["x", "a"], "lib_to_std": {"a": "a"}, "implicit_receiver": True}
  # obj should become x
  target_impl = {"std_args": ["x", "a"]}
  res = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args['x'] = obj
  assert len(res) == 2

  # Now let's test where it IS provided
  node2 = cst.parse_expression("obj.func(x=2, a=1)")
  assert isinstance(node2, cst.Call)
  res2 = normalize_arguments(node2, node2, mapping, target_impl, "torch", lambda x: False)
  # First std arg provided, so no implicit receiver injection
  assert len(res2) == 2

  # Now implicit_receiver = False and it is an Attribute
  mapping3 = {"std_args": ["x"], "implicit_receiver": False}
  node3 = cst.parse_expression("obj.func(x=1)")
  assert isinstance(node3, cst.Call)
  res3 = normalize_arguments(node3, node3, mapping3, target_impl, "torch", lambda x: False)
  assert len(res3) == 1


def test_normalize_args_positional_duplicate() -> None:
  """Verify `normalize_arguments` handles normalize args positional duplicate as expected."""
  import libcst as cst

  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  node = cst.parse_expression("func(1, 2)")
  assert isinstance(node, cst.Call)
  mapping = {"std_args": ["x", "x"], "implicit_receiver": False}
  target_impl = {"std_args": ["x"]}
  res = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args will have 'x': 1, and the second 'x' will be ignored but pos_idx increments
  assert len(res) == 2


def test_normalize_args_empty_variadic() -> None:
  """Verify `normalize_arguments` returns `0` for normalize args empty variadic."""
  import libcst as cst

  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  node = cst.parse_expression("func()")
  assert isinstance(node, cst.Call)
  mapping = {"std_args": [{"name": "*args", "is_variadic": True}], "implicit_receiver": False}
  target_impl = {"std_args": [{"name": "args"}], "pack_to_tuple": "args", "pack_as": "Tuple"}
  res = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args will NOT have '*args' because packing_mode was not entered
  assert len(res) == 0


def test_normalize_arguments_full() -> None:
  """Verify `Call` handles normalize arguments full as expected."""
  original = cst.Call(
    func=cst.Name("foo"),
    args=[
      cst.Arg(value=cst.Name("val1")),
      cst.Arg(keyword=cst.Name("k1"), value=cst.Name("v1")),
      cst.Arg(keyword=cst.Name("k_extra"), value=cst.Name("v_extra")),
    ],
  )

  target_cfg: dict[str, RuntimeTensorOrValue] = {
    "arg_values": {"arg1": "new_def1", "arg2": {"v1": "target_v1"}},
    "kwargs_map": {"k_extra": None, "arg3": "new_arg3"},
    "inject_args": {"injected1": "inj_val"},
  }
  config: dict[str, RuntimeTensorOrValue] = {
    "signature": {"args": [{"name": "arg1", "default": "def1"}, {"name": "arg2"}, ("arg3", "int"), "arg4"]},
    "library_to_std_args": {"k1": "arg2"},
    "target": target_cfg,
  }

  normalized = normalize_arguments(
    original,
    original,
    config,
    target_impl=target_cfg,
    source_framework="torch",
    is_module_alias_fn=lambda x: False,
  )
  assert len(normalized) > 0


def test_value_to_cst_negative_float() -> None:
  """Verify `convert_value_to_cst` handles value to CST negative float as expected."""
  res = convert_value_to_cst(-3.14)
  assert isinstance(res, cst.UnaryOperation)
  assert isinstance(res.operator, cst.Minus)
  assert isinstance(res.expression, cst.Float)
  assert res.expression.value == "3.14"
  assert cst.Module([]).code_for_node(res) == "-3.14"


def test_normalize_arguments_tuple_item() -> None:
  """Verify `Call` returns `1` for normalize arguments tuple item."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a"))])
  config = {
    "signature": {"args": [("my_arg", "int"), {"name": "my_opt_arg", "default": 42}]},
    "library_to_std_args": {},
  }
  normalized = normalize_arguments(
    original, original, config, target_impl={}, source_framework="torch", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) == 1


def test_normalize_arguments_method_call() -> None:
  """Verify `Call` returns `2` for normalize arguments method call."""
  original = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("foo")), args=[cst.Arg(value=cst.Name("a"))])
  config = {"signature": {"args": ["self", "arg1"]}, "library_to_std_args": {}}

  res = normalize_arguments(
    original, original, config, target_impl={}, source_framework="torch", is_module_alias_fn=lambda x: False
  )
  assert isinstance(res, list)
  assert len(res) == 2


def test_normalize_arguments_pack_variadic() -> None:
  """Verify `Call` handles normalize arguments pack variadic as expected."""
  original = cst.Call(
    func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a")), cst.Arg(value=cst.Name("b")), cst.Arg(value=cst.Name("c"))]
  )
  config = {
    "signature": {
      "args": [{"name": "var_args", "is_variadic": True}],
      "pack_variadic_into_keyword": "packed_args",
      "pack_variadic_type": "List",
    },
    "library_to_std_args": {},
  }
  res = normalize_arguments(
    original, original, config, target_impl={}, source_framework="torch", is_module_alias_fn=lambda x: False
  )
  assert isinstance(res, list)
  assert len(res) == 3
  assert cst.Module([]).code_for_node(cst.Call(func=cst.Name("foo"), args=res)) == "foo(a, b, c)"

  packed = normalize_arguments(
    original,
    original,
    {"std_args": [{"name": "var_args", "is_variadic": True}]},
    target_impl={"pack_to_tuple": "packed_args", "pack_as": "List"},
    source_framework="torch",
    is_module_alias_fn=lambda x: False,
  )
  assert len(packed) == 1
  assert cst.Module([]).code_for_node(cst.Call(func=cst.Name("foo"), args=packed)) == "foo(packed_args=[a, b, c])"


def test_normalize_arguments_pack_variadic_tuple() -> None:
  """Verify `Call` handles normalize arguments pack variadic tuple as expected."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a"))])
  config = {
    "signature": {
      "args": [{"name": "var_args", "is_variadic": True}],
      "pack_variadic_into_keyword": "packed_args",
      "pack_variadic_type": "Tuple",
    },
    "library_to_std_args": {},
  }
  res = normalize_arguments(
    original, original, config, target_impl={}, source_framework="torch", is_module_alias_fn=lambda x: False
  )
  assert isinstance(res, list)
  assert len(res) == 1
  assert cst.ensure_type(res[0].value, cst.Name).value == "a"

  two_args = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a")), cst.Arg(value=cst.Name("b"))])
  packed = normalize_arguments(
    two_args,
    two_args,
    {"std_args": [{"name": "var_args", "is_variadic": True}]},
    target_impl={"pack_to_tuple": "packed_args", "pack_as": "Tuple"},
    source_framework="torch",
    is_module_alias_fn=lambda x: False,
  )
  assert len(packed) == 1
  assert cst.Module([]).code_for_node(cst.Call(func=cst.Name("foo"), args=packed)) == "foo(packed_args=(a, b))"


def test_convert_value_negative_int() -> None:
  """Verify `convert_value_to_cst` handles convert value negative int as expected."""
  # Covers line 111
  node = convert_value_to_cst(-42)
  assert isinstance(node, cst.UnaryOperation)
  assert isinstance(node.operator, cst.Minus)
  assert isinstance(node.expression, cst.Integer)
  assert node.expression.value == "42"
  assert cst.Module([]).code_for_node(node) == "-42"


def test_normalize_arguments_various() -> None:
  """Verify `normalize_arguments` handles various as expected."""
  # Setup standard call
  original_node = cst.parse_expression("obj.func(1, 2, kw=3, bad=4)")
  assert isinstance(original_node, cst.Call)
  updated_node = original_node

  op_details = {
    "std_args": [
      {"name": ""},  # Empty name, covers 164->161
      {"name": "x", "is_variadic": False},  # Covers 166->168
      {"name": "y", "default": 10},  # Covers 169
      ["z", "z_alias"],  # Covers 171
      {"name": "v", "is_variadic": True},
    ],
    "variants": {"source_framework": {"args": {"kw": "z"}}},
  }

  target_impl = {
    "args": {"y": None},  # Covers 304
    "pack_to_tuple": "packed_v",
    "pack_as": "List",
  }

  def is_module_alias_fn(node: cst.CSTNode) -> bool:
    """Execute `is_module_alias_fn` helper in rewriter normalization utils tests.

    Args:
        node: CST or AST node under inspection or transformation.

    Returns:
        Resulting `bool` from is module alias fn.
    """
    return True  # Covers 196

  # Covers 196 (is_module_alias_fn returns True, so is_method_call = False)
  res = normalize_arguments(original_node, updated_node, op_details, target_impl, "source_framework", is_module_alias_fn)
  assert isinstance(res, list)
  assert len(res) >= 1


def test_normalize_arguments_packing() -> None:
  """Verify `normalize_arguments` handles packing as expected."""
  # covers 203->202, 226, 230-231, 248-265, 279-291, 297-298, 313-316, 324-329, 345-346, 380->383
  original_node = cst.parse_expression("func(1, 2, 3, v1=4, v2='val')")
  assert isinstance(original_node, cst.Call)
  updated_node = cst.parse_expression("func(1, 2, 3, 4)")
  assert isinstance(updated_node, cst.Call)

  op_details = {
    "std_args": [
      {"name": "a"},
      {"name": "v", "is_variadic": True},
      {"name": "missing", "default": 100},  # covers 279-291
    ]
  }

  target_impl = {
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

  res = normalize_arguments(original_node, updated_node, op_details, target_impl, "source_framework", lambda x: False)
  assert isinstance(res, list)
  assert len(res) >= 2


def test_normalize_arguments_missing_branches() -> None:
  """Verify `normalize_arguments` handles missing branches as expected."""
  # 203->202: is_method_call with positional args
  # 260: pack_as="Tuple" with exactly 1 element
  # 290-291: Exception in default value (invalid cst.Name)
  # 329: target_val_map with non-string value
  # 380->383: new_args_list[-1].comma == MaybeSentinel.DEFAULT

  original_node = cst.parse_expression("obj.func(1)")
  assert isinstance(original_node, cst.Call)
  updated_node = cst.parse_expression("func(1)")
  assert isinstance(updated_node, cst.Call)

  op_details = {
    "std_args": [
      {"name": "first"},
      {"name": "v", "is_variadic": True},
      {"name": "invalid-name", "default": 99},  # covers 290-291
    ]
  }

  target_impl = {
    "pack_to_tuple": "v_packed",
    "pack_as": "Tuple",  # covers 260
    "arg_values": {
      "first": 42,  # covers 329
    },
    "inject_args": {
      "inj1": "55",  # covers 380->381
    },
  }

  res = normalize_arguments(original_node, updated_node, op_details, target_impl, "source_framework", lambda x: False)
  assert isinstance(res, list)
  assert len(res) >= 2


def test_normalize_arguments_empty_list_injection() -> None:
  """Verify `normalize_arguments` returns `1` for empty list injection."""
  # Covers 380->383 where len(new_args_list) == 0
  original_node = cst.parse_expression("func()")
  assert isinstance(original_node, cst.Call)
  updated_node = cst.parse_expression("func()")
  assert isinstance(updated_node, cst.Call)

  op_details: dict[str, RuntimeTensorOrValue] = {"std_args": []}

  target_impl = {
    "inject_args": {
      "inj1": "55",
    }
  }

  res = normalize_arguments(original_node, updated_node, op_details, target_impl, "source_framework", lambda x: False)
  assert len(res) == 1


def test_convert_value_to_cst_ast_literal_str() -> None:
  """Verify `convert_value_to_cst` handles AST literal str as expected."""
  # If it parses to something else
  res = convert_value_to_cst("['a']")
  assert isinstance(res, cst.List)
  assert len(res.elements) == 1
  elem_val = res.elements[0].value
  assert isinstance(elem_val, cst.SimpleString)
  assert elem_val.value.strip("'\"") == "a"


def test_normalize_arguments_dict_variadic_default() -> None:
  """Verify `Call` handles normalize arguments dict variadic default as expected."""
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
    original, original, config, target_impl={}, source_framework="torch", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) == 0


def test_normalize_arguments_pack_variadic_one_element() -> None:
  """Verify `Call` handles normalize arguments pack variadic one element as expected."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a"))])
  config = {
    "signature": {
      "args": [{"name": "var_args", "is_variadic": True}],
      "pack_variadic_into_keyword": "packed_args",
      "pack_variadic_type": "Tuple",
    },
    "library_to_std_args": {},
  }
  res = normalize_arguments(
    original, original, config, target_impl={}, source_framework="torch", is_module_alias_fn=lambda x: False
  )
  assert isinstance(res, list)
  assert len(res) == 1
  assert cst.Module([]).code_for_node(cst.Call(func=cst.Name("foo"), args=res)) == "foo(a)"


def test_normalize_arguments_pack_variadic_list() -> None:
  """Verify `Call` handles normalize arguments pack variadic list as expected."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("a"))])
  config = {
    "signature": {
      "args": [{"name": "var_args", "is_variadic": True}],
      "pack_variadic_into_keyword": "packed_args",
      "pack_variadic_type": "List",
    },
    "library_to_std_args": {},
  }
  res = normalize_arguments(
    original, original, config, target_impl={}, source_framework="torch", is_module_alias_fn=lambda x: False
  )
  assert isinstance(res, list)
  assert len(res) == 1
  assert cst.Module([]).code_for_node(cst.Call(func=cst.Name("foo"), args=res)) == "foo(a)"


def test_normalize_arguments_method_call_module_alias() -> None:
  """Verify `Call` handles normalize arguments method call module alias as expected."""
  original = cst.Call(func=cst.Attribute(value=cst.Name("np"), attr=cst.Name("foo")), args=[cst.Arg(value=cst.Name("a"))])
  config = {"signature": {"args": ["self", "arg1"]}, "library_to_std_args": {}}

  normalized = normalize_arguments(
    original, original, config, target_impl={}, source_framework="numpy", is_module_alias_fn=lambda x: x == "np"
  )
  # np.foo(a) won't have self injected. So 2 arg in result actually, because receiver injection adds 'np'
  assert len(normalized) == 2


def test_normalize_arguments_method_call_no_std_args_order() -> None:
  """Verify `Call` handles normalize arguments method call no std args order as expected."""
  original = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("foo")),
    args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))],
  )
  config: dict[str, RuntimeTensorOrValue] = {"signature": {"args": []}, "library_to_std_args": {}}
  res = normalize_arguments(
    original, original, config, target_impl={}, source_framework="numpy", is_module_alias_fn=lambda x: False
  )
  assert isinstance(res, list)
  assert len(res) == 2


def test_normalize_arguments_inject_defaults_false_and_bad_cst() -> None:
  """Verify inject_defaults=False and BadCST repr exception handling."""
  from unittest.mock import patch

  call = cst.Call(func=cst.Name("func"), args=[])
  details = {"std_args": [{"name": "a", "default": 5}]}
  target_impl = {"args": {"a": "x"}, "inject_defaults": False}
  assert normalize_arguments(call, call, details, target_impl, "torch", lambda x: False) == []

  class BadCST:
    """Sentinel `BadCST` fixture class used in rewriter normalization utils tests."""

    def __repr__(self) -> str:
      """Raise RuntimeError on repr.

      Returns:
          Resulting `str` from repr.
      """
      raise RuntimeError("fail")

  details_bad = {"std_args": [{"name": "a", "default": BadCST()}]}
  assert normalize_arguments(call, call, details_bad, {"args": {"a": "a"}}, "torch", lambda x: False) == []

  with patch("ast.literal_eval", return_value="foo"):
    node = convert_value_to_cst("foo")
    assert isinstance(node, cst.SimpleString)


def test_normalize_arguments_extra_args_kwargs_map() -> None:
  """Verify `Call` handles normalize arguments extra args kwargs map as expected."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))])
  config: dict[str, RuntimeTensorOrValue] = {"signature": {"args": []}, "library_to_std_args": {}}
  target_impl = {"kwargs_map": {"kw": "new_kw"}}
  normalized = normalize_arguments(
    original, original, config, target_impl=target_impl, source_framework="numpy", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) == 1


def test_normalize_arguments_value_options_str_syntax_error() -> None:
  """Verify `Call` raises `ParserSyntaxError` during normalize arguments value options str syntax error."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))])
  config = {"signature": {"args": ["kw"]}, "library_to_std_args": {}}
  target_impl = {"inject_args": {"kw": "invalid syntax!@"}}
  # It catches exception? actually inject args calls parse_expression! Let's just catch it in the test
  import pytest
  from libcst import ParserSyntaxError

  with pytest.raises(ParserSyntaxError):
    normalize_arguments(
      original, original, config, target_impl=target_impl, source_framework="numpy", is_module_alias_fn=lambda x: False
    )


def test_normalize_arguments_value_options_complex_obj() -> None:
  """Verify `Call` handles normalize arguments value options complex obj as expected."""
  original = cst.Call(func=cst.Name("foo"), args=[cst.Arg(keyword=cst.Name("kw"), value=cst.Name("a"))])
  config = {"signature": {"args": ["kw"]}, "library_to_std_args": {}}
  target_impl = {"arg_values": {"kw": {"some": "dict", "complex": "value"}}}
  normalized = normalize_arguments(
    original, original, config, target_impl=target_impl, source_framework="numpy", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) == 1


def test_convert_value_to_cst_literal_eval_same_str() -> None:
  """Verify `convert_value_to_cst` unwraps quoted Python string literals into `cst.SimpleString`."""
  res = convert_value_to_cst("'test'")
  assert isinstance(res, cst.SimpleString)
  assert res.evaluated_value == "test"


def test_normalize_arguments_dynamic_receiver_attribute_check() -> None:
  """Test normalize_arguments when original_node.func ceases to be an attribute mid-processing."""
  counts: dict[int, int] = {}
  func_pairs: dict[int, tuple[cst.BaseExpression, cst.BaseExpression]] = {}

  class _DynamicFuncCall(cst.Call):
    """Mock `cst.Call` subclass with dynamic `func` property."""

    def __init__(self, first_func: cst.BaseExpression, second_func: cst.BaseExpression, args: list[cst.Arg]) -> None:
      """Initialize mock call.

      Args:
          first_func: Input `first_func` parameter value (`cst.BaseExpression`).
          second_func: Input `second_func` parameter value (`cst.BaseExpression`).
          args: Positional argument specifications (`list[cst.Arg]`).
      """
      counts[id(self)] = 0
      func_pairs[id(self)] = (first_func, second_func)
      super().__init__(func=first_func, args=args)

    @property
    def func(self) -> cst.BaseExpression:
      """Return first_func on initial checks, then second_func.

      Returns:
          Resulting `cst.BaseExpression` from func.
      """
      key = id(self)
      counts[key] += 1
      first_fn, second_fn = func_pairs[key]
      if counts[key] <= 2:
        return first_fn
      return second_fn

    @func.setter
    def func(self, value: cst.BaseExpression) -> None:
      """Ignore dataclass init assignment to `func` so property remains active.

      Args:
          value: Assigned `cst.BaseExpression` from superclass initialization.
      """
      _ = value

  # Branch 211->220: std_args_order present, but func becomes non-attribute
  mock_call = _DynamicFuncCall(
    cst.Attribute(value=cst.Name("obj"), attr=cst.Name("method")),
    cst.Name("not_attr"),
    [cst.Arg(value=cst.Name("val"))],
  )
  config: dict[str, RuntimeTensorOrValue] = {"std_args": ["first", "second"], "library_to_std_args": {}}
  res1 = normalize_arguments(
    mock_call,
    cst.Call(func=cst.Name("method"), args=[cst.Arg(value=cst.Name("val"))]),
    config,
    target_impl={},
    source_framework="torch",
    is_module_alias_fn=lambda _: False,
  )
  assert len(res1) == 1

  # Branch 216->220: std_args_order empty, but func becomes non-attribute
  mock_call_no_std = _DynamicFuncCall(
    cst.Attribute(value=cst.Name("obj"), attr=cst.Name("method")),
    cst.Name("not_attr"),
    [cst.Arg(value=cst.Name("val"))],
  )
  config_no_std: dict[str, RuntimeTensorOrValue] = {"std_args": [], "library_to_std_args": {}}
  res2 = normalize_arguments(
    mock_call_no_std,
    cst.Call(func=cst.Name("method"), args=[cst.Arg(value=cst.Name("val"))]),
    config_no_std,
    target_impl={},
    source_framework="torch",
    is_module_alias_fn=lambda _: False,
  )
  assert len(res2) == 1


def test_normalize_arguments_val_options_dict_unmatched_key() -> None:
  """Test val_options dictionary where primitive key is not present in dictionary."""
  call = cst.Call(func=cst.Name("foo"), args=[cst.Arg(keyword=cst.Name("mode"), value=cst.SimpleString("'unmatched'"))])
  config = {"std_args": ["mode"], "library_to_std_args": {}}
  target_impl = {"arg_values": {"mode": {"known": "1"}}}
  res = normalize_arguments(
    call,
    call,
    config,
    target_impl=target_impl,
    source_framework="torch",
    is_module_alias_fn=lambda _: False,
  )
  assert len(res) == 1
