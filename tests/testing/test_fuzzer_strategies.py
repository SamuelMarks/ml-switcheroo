"""Test suite for the Fuzzer Strategies module."""

from ml_switcheroo.testing.fuzzer.type_parser import parse_type_annotation


def test_get_dtype_strategy():
  """Gets dtype strategy."""
  from ml_switcheroo.testing.fuzzer.strategies import _get_dtype_strategy
  import numpy as np

  assert _get_dtype_strategy(None) == np.float32
  assert _get_dtype_strategy("int") == np.int32
  assert _get_dtype_strategy("int64") == np.int64
  assert _get_dtype_strategy("long") == np.int64
  assert _get_dtype_strategy("float") == np.float32
  assert _get_dtype_strategy("float32") == np.float32
  assert _get_dtype_strategy("float64") == np.float64
  assert _get_dtype_strategy("double") == np.float64
  assert _get_dtype_strategy("bool") is bool
  assert _get_dtype_strategy("int16") == np.int16
  assert _get_dtype_strategy("invalid_dtype") == np.float32


def test_strategies_from_spec():
  """Verifies the behavior of strategies from spec."""
  from ml_switcheroo.testing.fuzzer.strategies import strategies_from_spec

  strategies_from_spec("int", {"options": [1, 2]})
  strategies_from_spec("int | float", {})
  strategies_from_spec("int", {"min": 0, "max": 10})
  strategies_from_spec("float", {})
  strategies_from_spec("bool", {})
  strategies_from_spec("str", {})
  strategies_from_spec("Callable", {})
  strategies_from_spec("Optional[int]", {})
  strategies_from_spec("List[int]", {})
  strategies_from_spec("Tuple[int, ...]", {})
  strategies_from_spec("Tuple[int, float]", {})
  strategies_from_spec("Dict[str, int]", {})
  strategies_from_spec("Dict[List[int], int]", {})
  strategies_from_spec("dtype", {})
  strategies_from_spec("unknown_type", {"default": 42})
  strategies_from_spec("unknown_type", {})


def test_array_strategy():
  """Verifies the behavior of array strategy."""
  from ml_switcheroo.testing.fuzzer.strategies import _array_strategy

  shared_dims = {}
  _array_strategy(parse_type_annotation("Array['N', 32, 'M']"), {}, shared_dims)
  _array_strategy(parse_type_annotation("Array['8']"), {}, shared_dims)
  _array_strategy(parse_type_annotation("Array"), {"min": 5, "max": 10, "dtype": "int", "rank": 2}, {})
  _array_strategy(parse_type_annotation("Array['N+1']"), {}, shared_dims)
  _array_strategy(parse_type_annotation("Array"), {"min": 5, "max": 10, "dtype": "int", "rank": 2}, {})


def test_strategies_from_spec_more():
  """Verifies the behavior of strategies from spec more."""
  from ml_switcheroo.testing.fuzzer.strategies import strategies_from_spec, _array_strategy

  strategies_from_spec("List[int] | float", {})
  strategies_from_spec("Array['N']", {})
  strategies_from_spec("Tensor['N']", {})
  strategies_from_spec("np.ndarray", {})
  strategies_from_spec("None", {})
  _array_strategy(parse_type_annotation("Array"), {"min": 5, "max": 10, "dtype": "int", "rank": 2}, {})
