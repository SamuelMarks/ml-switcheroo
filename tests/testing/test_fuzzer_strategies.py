"""Test suite for the Fuzzer Strategies module."""

import typing

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from ml_switcheroo.testing.fuzzer.type_parser import parse_type_annotation

_T = typing.TypeVar("_T")


def _draw_one(strategy: st.SearchStrategy[_T]) -> _T:
  """Draw a single value from a Hypothesis strategy without interactive warnings.

  Args:
      strategy: Hypothesis search strategy to sample.

  Returns:
      Sampled value produced by `strategy`.
  """
  samples: list[_T] = []

  @given(val=strategy)
  @settings(max_examples=1, database=None)
  def _collect(val: _T) -> None:
    """Append one sampled value from Hypothesis.

    Args:
        val: Generated strategy value.
    """
    samples.append(val)

  _collect()
  return samples[0]


def test_get_dtype_strategy() -> None:
  """Gets dtype strategy."""
  from ml_switcheroo.testing.fuzzer.strategies import _get_dtype_strategy

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


def test_strategies_from_spec() -> None:
  """Verify `strategies_from_spec` produces expected results across fuzzer strategies inputs."""
  from ml_switcheroo.testing.fuzzer.strategies import strategies_from_spec

  opt_strat = strategies_from_spec("int", {"options": [1, 2]})
  assert _draw_one(opt_strat) in (1, 2)
  strategies_from_spec("int | float", {})
  int_strat = strategies_from_spec("int", {"min": 0, "max": 10})
  int_val = _draw_one(int_strat)
  assert isinstance(int_val, int)
  assert int_val >= 0
  assert int_val <= 10
  strategies_from_spec("float", {})
  strategies_from_spec("bool", {})
  strategies_from_spec("str", {})
  call_strat = strategies_from_spec("Callable", {})
  fn = _draw_one(call_strat)
  assert callable(fn)
  assert fn(99) == 99
  strategies_from_spec("Optional[int]", {})
  strategies_from_spec("List[int]", {})
  strategies_from_spec("Tuple[int, ...]", {})
  tup_strat = strategies_from_spec("Tuple[int, float]", {})
  tup_val = _draw_one(tup_strat)
  assert isinstance(tup_val, tuple)
  assert len(tup_val) == 2
  strategies_from_spec("Dict[str, int]", {})
  strategies_from_spec("Dict[List[int], int]", {})
  strategies_from_spec("dtype", {})
  def_strat = strategies_from_spec("unknown_type", {"default": 42})
  assert _draw_one(def_strat) == 42
  _res = strategies_from_spec("unknown_type", {})
  assert isinstance(_res, st.SearchStrategy)
  arr_val = _draw_one(_res)
  assert isinstance(arr_val, np.ndarray)
  assert arr_val.dtype == np.float32
  assert arr_val.ndim >= 1


def test_array_strategy() -> None:
  """Verify `_array_strategy` produces expected results across fuzzer strategies inputs."""
  from ml_switcheroo.testing.fuzzer.strategies import _array_strategy

  shared_dims: dict[str, st.SearchStrategy[int]] = {}
  sym_strat = _array_strategy(
    parse_type_annotation("Array['N', 32, 'M']"),
    {},
    shared_dims,
  )
  assert set(shared_dims.keys()) == {"N", "M"}
  sym_arr = _draw_one(sym_strat)
  assert sym_arr.ndim == 2

  fixed_strat = _array_strategy(
    parse_type_annotation("Array['8']"),
    {},
    shared_dims,
  )
  assert _draw_one(fixed_strat).shape == (8,)

  int_strat = _array_strategy(parse_type_annotation("Array"), {"min": 5, "max": 10, "dtype": "int", "rank": 2}, {})
  int_arr = _draw_one(int_strat)
  assert int_arr.ndim == 2
  assert int_arr.dtype == np.int32
  assert bool(np.all(int_arr >= 5)) is True
  assert bool(np.all(int_arr <= 10)) is True

  _array_strategy(
    parse_type_annotation("Array['N+1']"),
    {},
    shared_dims,
  )
  _array_strategy(parse_type_annotation("Array"), {"min": 5, "max": 10, "dtype": "int", "rank": 2}, {})
  shared_dims_dup: dict[str, st.SearchStrategy[int]] = {}
  dup_strat = _array_strategy(
    parse_type_annotation("Array['N', 'N']"),
    {},
    shared_dims_dup,
  )
  assert list(shared_dims_dup.keys()) == ["N"]
  dup_arr = _draw_one(dup_strat)
  assert dup_arr.shape[0] == dup_arr.shape[1]

  _res = _array_strategy(parse_type_annotation("Array"), {"dtype": "bool", "rank": 2}, {})
  assert isinstance(_res, st.SearchStrategy)
  bool_arr = _draw_one(_res)
  assert bool_arr.ndim == 2
  assert bool_arr.dtype == np.bool_


def test_strategies_from_spec_more() -> None:
  """Verify `strategies_from_spec` handles more as expected."""
  from ml_switcheroo.testing.fuzzer.strategies import _array_strategy, strategies_from_spec

  strategies_from_spec("List[int] | float", {})
  arr_n_strat = strategies_from_spec("Array['N']", {})
  arr_n = _draw_one(arr_n_strat)
  assert isinstance(arr_n, np.ndarray)
  assert arr_n.ndim == 1
  strategies_from_spec("Tensor['N']", {})
  strategies_from_spec("np.ndarray", {})
  none_strat = strategies_from_spec("None", {})
  assert _draw_one(none_strat) is None
  _res = _array_strategy(parse_type_annotation("Array"), {"min": 5, "max": 10, "dtype": "int", "rank": 2}, {})
  assert isinstance(_res, st.SearchStrategy)
  res_arr = _draw_one(_res)
  assert res_arr.ndim == 2
  assert res_arr.dtype == np.int32
  assert bool(np.all(res_arr >= 5)) is True
  assert bool(np.all(res_arr <= 10)) is True
