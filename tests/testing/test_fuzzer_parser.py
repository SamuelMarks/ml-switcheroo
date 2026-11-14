"""Test suite for the Fuzzer Parser module."""

from ml_switcheroo.testing.fuzzer.type_parser import (
  CallableType,
  DictType,
  ListType,
  PrimitiveType,
  TupleType,
  parse_type_annotation,
)


def test_get_fallback_base_value() -> None:
  """Gets fallback base value."""
  import numpy as np

  from ml_switcheroo.testing.fuzzer.parser import get_fallback_base_value

  assert get_fallback_base_value(PrimitiveType(name="bool"), ()) is False
  assert get_fallback_base_value(PrimitiveType(name="int"), ()) == 0
  assert get_fallback_base_value(parse_type_annotation("integer"), ()) == 0
  assert get_fallback_base_value(PrimitiveType(name="float"), ()) == 0.0
  assert get_fallback_base_value(PrimitiveType(name="str"), ()) == ""
  assert get_fallback_base_value(PrimitiveType(name="dtype"), ()) == np.float32
  fb_arr = get_fallback_base_value(parse_type_annotation("Array"), (2,))
  assert isinstance(fb_arr, np.ndarray)
  assert fb_arr.shape == (2,)
  assert get_fallback_base_value(ListType(inner=PrimitiveType(name="int")), ()) == []
  assert get_fallback_base_value(TupleType(elements=[PrimitiveType(name="int")], variadic=False), ()) == ()
  assert (
    get_fallback_base_value(DictType(key_type=PrimitiveType(name="int"), value_type=PrimitiveType(name="int")), ()) == {}
  )
  assert get_fallback_base_value(CallableType(), ()) is not None
  assert get_fallback_base_value(parse_type_annotation("unknown"), ()) is None


def test_generate_from_hint() -> None:
  """Generates from hint."""
  import numpy as np

  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint

  symbol_map: dict[str, int] = {}
  assert generate_from_hint("int", (), 10, 5, symbol_map) == 0
  assert generate_from_hint("int", (), 0, 5, symbol_map, {"options": [42]}) == 42
  res = generate_from_hint("Any", (), 0, 5, symbol_map, {"default": 42})
  assert isinstance(res, int)
  res = generate_from_hint("Any", (), 0, 5, symbol_map, {"default": 42.0})
  assert isinstance(res, float)
  res = generate_from_hint("Any", (), 0, 5, symbol_map, {"default": True})
  assert isinstance(res, bool)
  res = generate_from_hint("Any", (), 0, 5, symbol_map, {"default": [1]})
  assert isinstance(res, list)
  res = generate_from_hint("Any", (), 0, 5, symbol_map, {"default": [1.0]})
  assert isinstance(res, list)
  res = generate_from_hint("int | float", (), 0, 5, symbol_map)
  assert isinstance(res, (int, float))
  res = generate_from_hint("Optional[int]", (), 0, 5, symbol_map)
  assert res is None or isinstance(res, int)
  res = generate_from_hint("Tuple[int, ...]", (), 0, 5, symbol_map)
  assert isinstance(res, tuple)
  res = generate_from_hint("Tuple[int, float]", (), 0, 5, symbol_map)
  assert isinstance(res, tuple)
  assert len(res) == 2
  assert isinstance(res[0], int)
  assert isinstance(res[1], float)
  res = generate_from_hint("List[Array['N']]", (2,), 0, 5, symbol_map)
  assert isinstance(res, list)
  assert len(res) >= 2
  first_elem = res[0]
  assert isinstance(first_elem, np.ndarray)
  for elem in res:
    assert isinstance(elem, np.ndarray)
    assert elem.shape == first_elem.shape
  res = generate_from_hint("List[int]", (), 0, 5, symbol_map)
  assert isinstance(res, list)
  res = generate_from_hint("Dict[str, int]", (), 0, 5, symbol_map)
  assert isinstance(res, dict)
  assert generate_from_hint("None", (), 0, 5, symbol_map) is None
  res = generate_from_hint("Array['N', 32]", (), 0, 5, symbol_map, {"rank": 2})
  assert isinstance(res, np.ndarray)
  res = generate_from_hint("Array", (2,), 0, 5, symbol_map, {"rank": 3})
  assert isinstance(res, np.ndarray)
  assert len(res.shape) == 3
  res = generate_from_hint("Callable", (), 0, 5, symbol_map)
  assert callable(res)
  res = generate_from_hint("int", (), 0, 5, symbol_map)
  assert isinstance(res, int)
  res = generate_from_hint("float", (), 0, 5, symbol_map)
  assert isinstance(res, float)
  res = generate_from_hint("bool", (), 0, 5, symbol_map)
  assert isinstance(res, bool)
  res = generate_from_hint("str", (), 0, 5, symbol_map)
  assert isinstance(res, str)
  res = generate_from_hint("dtype", (), 0, 5, symbol_map)
  res = generate_from_hint("unknown_type_xxx", (2,), 0, 5, symbol_map)
  assert isinstance(res, np.ndarray)


def test_fuzzer_parser_missed() -> None:
  """Verify `generate_from_hint` handles fuzzer parser missed as expected."""
  from unittest import mock

  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint

  symbol_map: dict[str, int] = {}
  with mock.patch("random.random", return_value=0.1):
    assert generate_from_hint("Optional[int]", (), 0, 5, symbol_map) is None
  with mock.patch("ml_switcheroo.testing.fuzzer.parser.generate_from_hint") as mock_gen:
    import numpy as np

    mock_gen.side_effect = [np.zeros((2,)), np.zeros((3,))]
    with mock.patch("random.randint", return_value=2):
      res = generate_from_hint("List[Array]", (2,), 0, 5, symbol_map)
  res = generate_from_hint("Dict[List[int], int]", (), 0, 5, symbol_map)
  res = generate_from_hint("Dict[int]", (), 0, 5, symbol_map)
  assert res == {}


def test_fuzzer_parser_more() -> None:
  """Verify `generate_from_hint` handles fuzzer parser more as expected."""
  from unittest import mock

  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint

  symbol_map: dict[str, int] = {}
  with mock.patch("random.random", return_value=0.5):
    res1 = generate_from_hint("Any", (), 0, 5, symbol_map, {"default": [1.0]})
    res2 = generate_from_hint("Any", (), 0, 5, symbol_map, {"default": []})
  assert isinstance(res1, list)
  assert len(res1) in (2, 3)
  assert isinstance(res1[0], list)
  assert isinstance(res2, list)
  assert len(res2) in (2, 3)
  assert isinstance(res2[0], list)


def test_fuzzer_parser_bool_list() -> None:
  """Verify `generate_from_hint` handles fuzzer parser bool list as expected."""
  from unittest import mock

  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint

  symbol_map: dict[str, int] = {}
  with mock.patch("random.random", return_value=0.5):
    res_bool = generate_from_hint("Any", (), 0, 5, symbol_map, {"default": True})
    res_list = generate_from_hint("Any", (), 0, 5, symbol_map, {"default": [1, 2, 3]})
  assert isinstance(res_bool, bool)
  assert res_bool in (True, False)
  assert isinstance(res_list, list)
  assert len(res_list) in (2, 3)
  for item in res_list:
    assert isinstance(item, int)
    assert not isinstance(item, bool)
    assert item >= -5
    assert item <= 5


def test_fuzzer_parser_int_inference() -> None:
  """Verify `generate_from_hint` handles fuzzer parser int inference as expected."""
  from unittest import mock

  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint

  symbol_map: dict[str, int] = {}
  with mock.patch("random.random", return_value=0.5):
    res = generate_from_hint("Any", (), 0, 5, symbol_map, {"default": 42})
    bounded = generate_from_hint("Any", (), 0, 5, symbol_map, {"default": 42, "min": 7, "max": 7})
  assert isinstance(res, int)
  assert not isinstance(res, bool)
  assert res >= -5
  assert res <= 5
  assert bounded == 7


def test_fuzzer_parser_float_inference() -> None:
  """Verify `generate_from_hint` handles fuzzer parser float inference as expected."""
  from unittest import mock

  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint
  from ml_switcheroo.testing.fuzzer.type_parser import AnyType

  # Force the code to take the type inference branch instead of just returning default
  with mock.patch("random.random", return_value=0.5):
    val = generate_from_hint(AnyType(), (), 0, 3, {}, {"default": 3.14})
    assert isinstance(val, float)
    assert val == 0.5


def test_fuzzer_parser_additional_branches() -> None:
  """Verifies parser branch coverage for non-primitive defaults and primitive fallbacks."""
  from unittest import mock

  import numpy as np

  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint
  from ml_switcheroo.testing.fuzzer.type_parser import AnyType, PrimitiveType

  symbol_map: dict[str, int] = {}
  with mock.patch("random.random", return_value=0.5):
    # default_val is string (not bool, int, float, list) -> triggers 136->143
    res = generate_from_hint(AnyType(), (2, 2), 0, 5, symbol_map, {"default": "some_str"})
    assert isinstance(res, np.ndarray)
    assert res.shape == (2, 2)
    assert res.dtype == np.float32

  # PrimitiveType with unrecognized name -> triggers 224->238
  arr = generate_from_hint(PrimitiveType(name="unrecognized_primitive"), (2, 2), 0, 5, symbol_map)
  assert isinstance(arr, np.ndarray)
  assert arr.shape == (2, 2)
  assert arr.dtype == np.float32
