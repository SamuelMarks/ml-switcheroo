def test_get_dtype_strategy():
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
  assert _get_dtype_strategy("bool") == bool
  assert _get_dtype_strategy("int16") == np.int16
  assert _get_dtype_strategy("invalid_dtype") == np.float32


def test_strategies_from_spec():
  from ml_switcheroo.testing.fuzzer.strategies import strategies_from_spec
  from hypothesis import strategies as st

  # options
  s = strategies_from_spec("int", {"options": [1, 2]})

  # Union
  s = strategies_from_spec("int | float", {})

  # int
  s = strategies_from_spec("int", {"min": 0, "max": 10})

  # float
  s = strategies_from_spec("float", {})

  # bool
  s = strategies_from_spec("bool", {})

  # str
  s = strategies_from_spec("str", {})

  # Callable
  s = strategies_from_spec("Callable", {})

  # Optional
  s = strategies_from_spec("Optional[int]", {})

  # List
  s = strategies_from_spec("List[int]", {})

  # Tuple variable
  s = strategies_from_spec("Tuple[int, ...]", {})

  # Tuple fixed
  s = strategies_from_spec("Tuple[int, float]", {})

  # Dict
  s = strategies_from_spec("Dict[str, int]", {})

  # Dict with complex key
  s = strategies_from_spec("Dict[List[int], int]", {})

  # Dtype
  s = strategies_from_spec("dtype", {})

  # default fallback
  s = strategies_from_spec("unknown_type", {"default": 42})

  # Array fallback
  s = strategies_from_spec("unknown_type", {})


def test_array_strategy():
  from ml_switcheroo.testing.fuzzer.strategies import _array_strategy

  shared_dims = {}
  s = _array_strategy("Array['N', 32, 'M']", {}, shared_dims)

  s = _array_strategy("Array", {"rank": 2}, shared_dims)

  s = _array_strategy("Array['N+1']", {}, shared_dims)

  s = _array_strategy("Array", {}, shared_dims)


def test_strategies_from_spec_more():
  from ml_switcheroo.testing.fuzzer.strategies import strategies_from_spec, _array_strategy

  s = strategies_from_spec("List[int] | float", {})

  s = strategies_from_spec("Array['N']", {})
  s = strategies_from_spec("Tensor['N']", {})
  s = strategies_from_spec("np.ndarray", {})

  _array_strategy("Array", {"min": 5, "max": 10, "dtype": "int", "rank": 2}, {})
