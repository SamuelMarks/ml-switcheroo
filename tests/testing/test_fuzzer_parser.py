"""Test suite for the Fuzzer Parser module."""

from ml_switcheroo.testing.fuzzer.type_parser import (
  parse_type_annotation,
  PrimitiveType,
  ListType,
  TupleType,
  DictType,
  CallableType,
)


def test_get_fallback_base_value():
  """Gets fallback base value."""
  from ml_switcheroo.testing.fuzzer.parser import get_fallback_base_value

  assert get_fallback_base_value(PrimitiveType(name="bool"), ()) is False
  assert get_fallback_base_value(PrimitiveType(name="int"), ()) == 0
  assert get_fallback_base_value(parse_type_annotation("integer"), ()) == 0
  assert get_fallback_base_value(PrimitiveType(name="float"), ()) == 0.0
  assert get_fallback_base_value(PrimitiveType(name="str"), ()) == ""
  assert get_fallback_base_value(parse_type_annotation("Array"), (2,)).shape == (2,)
  assert get_fallback_base_value(ListType(inner=PrimitiveType(name="int")), ()) == []
  assert get_fallback_base_value(TupleType(elements=[PrimitiveType(name="int")], variadic=False), ()) == ()
  assert (
    get_fallback_base_value(DictType(key_type=PrimitiveType(name="int"), value_type=PrimitiveType(name="int")), ()) == {}
  )
  assert get_fallback_base_value(CallableType(), ()) is not None
  assert get_fallback_base_value(parse_type_annotation("unknown"), ()) is None


def test_generate_from_hint():
  """Generates from hint."""
  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint
  import numpy as np

  symbol_map = {}
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
  assert len(res) == 2
  assert isinstance(res[0], int)
  assert isinstance(res[1], float)
  res = generate_from_hint("List[Array['N']]", (2,), 0, 5, symbol_map)
  assert isinstance(res, list)
  assert len(res) >= 2
  assert all((isinstance(x, np.ndarray) for x in res))
  assert all((x.shape == res[0].shape for x in res))
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


def test_fuzzer_parser_missed():
  """Verifies the behavior of fuzzer parser missed."""
  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint

  symbol_map = {}
  with __import__("unittest.mock").mock.patch("random.random", return_value=0.1):
    assert generate_from_hint("Optional[int]", (), 0, 5, symbol_map) is None
  with __import__("unittest.mock").mock.patch("ml_switcheroo.testing.fuzzer.parser.generate_from_hint") as mock_gen:
    import numpy as np

    mock_gen.side_effect = [np.zeros((2,)), np.zeros((3,))]
    with __import__("unittest.mock").mock.patch("random.randint", return_value=2):
      res = generate_from_hint("List[Array]", (2,), 0, 5, symbol_map)
  res = generate_from_hint("Dict[List[int], int]", (), 0, 5, symbol_map)
  res = generate_from_hint("Dict[int]", (), 0, 5, symbol_map)
  assert res == {}


def test_fuzzer_parser_more():
  """Verifies the behavior of fuzzer parser more."""
  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint

  symbol_map = {}
  with __import__("unittest.mock").mock.patch("random.random", return_value=0.5):
    generate_from_hint("Any", (), 0, 5, symbol_map, {"default": [1.0]})
    generate_from_hint("Any", (), 0, 5, symbol_map, {"default": []})


def test_fuzzer_parser_bool_list():
  """Verifies the behavior of fuzzer parser boolean list."""
  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint

  symbol_map = {}
  with __import__("unittest.mock").mock.patch("random.random", return_value=0.5):
    generate_from_hint("Any", (), 0, 5, symbol_map, {"default": True})
    generate_from_hint("Any", (), 0, 5, symbol_map, {"default": [1, 2, 3]})


def test_fuzzer_parser_int_inference():
  """Verifies the behavior of fuzzer parser integer inference."""
  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint

  symbol_map = {}
  with __import__("unittest.mock").mock.patch("random.random", return_value=0.5):
    generate_from_hint("Any", (), 0, 5, symbol_map, {"default": 42})
