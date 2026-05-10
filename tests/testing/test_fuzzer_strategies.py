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
  assert _get_dtype_strategy("bool") is bool
  assert _get_dtype_strategy("int16") == np.int16
  assert _get_dtype_strategy("invalid_dtype") == np.float32


def test_strategies_from_spec():
  from ml_switcheroo.testing.fuzzer.strategies import strategies_from_spec

  # options
  strategies_from_spec("int", {"options": [1, 2]})

  # Union
  strategies_from_spec("int | float", {})

  # int
  strategies_from_spec("int", {"min": 0, "max": 10})

  # float
  strategies_from_spec("float", {})

  # bool
  strategies_from_spec("bool", {})

  # str
  strategies_from_spec("str", {})

  # Callable
  strategies_from_spec("Callable", {})

  # Optional
  strategies_from_spec("Optional[int]", {})

  # List
  strategies_from_spec("List[int]", {})

  # Tuple variable
  strategies_from_spec("Tuple[int, ...]", {})

  # Tuple fixed
  strategies_from_spec("Tuple[int, float]", {})

  # Dict
  strategies_from_spec("Dict[str, int]", {})

  # Dict with complex key
  strategies_from_spec("Dict[List[int], int]", {})

  # Dtype
  strategies_from_spec("dtype", {})

  # default fallback
  strategies_from_spec("unknown_type", {"default": 42})

  # Array fallback
  strategies_from_spec("unknown_type", {})


def test_array_strategy():
  from ml_switcheroo.testing.fuzzer.strategies import _array_strategy

  shared_dims = {}
  _array_strategy("Array['N', 32, 'M']", {}, shared_dims)

  _array_strategy("Array", {"rank": 2}, shared_dims)

  _array_strategy("Array['N+1']", {}, shared_dims)

  _array_strategy("Array", {}, shared_dims)


def test_strategies_from_spec_more():
  from ml_switcheroo.testing.fuzzer.strategies import strategies_from_spec, _array_strategy

  strategies_from_spec("List[int] | float", {})

  strategies_from_spec("Array['N']", {})
  strategies_from_spec("Tensor['N']", {})
  strategies_from_spec("np.ndarray", {})

  _array_strategy("Array", {"min": 5, "max": 10, "dtype": "int", "rank": 2}, {})
