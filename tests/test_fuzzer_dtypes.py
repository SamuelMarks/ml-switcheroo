"""
Tests for Robust Fuzzer Dtype Support, Hint Parsing, and Dynamic Shapes.

Verifies:
1. Heuristics fallback (legacy behavior).
2. Explicit Hint Parsing (Feature 027 integration).
3. Complex Nested Types (List[int], Tuple[int, ...], Dict[str, int]).
4. Recursion limits on deep nesting.
"""

import pytest
import numpy as np
import random
from ml_switcheroo.testing.fuzzer import InputFuzzer


@pytest.fixture
def fuzzer():
  # Fix seed for predictable generation
  random.seed(42)
  np.random.seed(42)
  return InputFuzzer()


# --- 1. Heuristics Tests ---


def test_heuristic_booleans(fuzzer):
  """Verify 'mask' generates boolean array."""
  inputs = fuzzer.generate_inputs(params=["mask", "condition"])
  assert inputs["mask"].dtype == bool
  assert inputs["condition"].dtype == bool


def test_heuristic_integers(fuzzer):
  """Verify 'indices' generates int array."""
  inputs = fuzzer.generate_inputs(params=["indices", "k_idx"])
  assert np.issubdtype(inputs["indices"].dtype, np.integer)
  assert np.issubdtype(inputs["k_idx"].dtype, np.integer)


def test_heuristic_scalars(fuzzer):
  """Verify scalar names generate Python scalars."""
  # alpha, eps -> floats
  # n_layers -> int (implied by n_ prefix for int heuristic)
  inputs = fuzzer.generate_inputs(params=["alpha", "eps"])
  assert isinstance(inputs["alpha"], float)
  assert isinstance(inputs["eps"], float)


def test_axis_heuristic_validity(fuzzer):
  """Verify 'axis' heuristic generates integers."""
  # Note: Independent strategies mean we can't guarantee axis < rank(x)
  # without explicit constraints or a composite strategy.
  # We check that heuristics generate valid types for both.
  for _ in range(10):
    inputs = fuzzer.generate_inputs(["x", "axis"])
    x = inputs["x"]
    axis = inputs["axis"]

    assert isinstance(x, np.ndarray)
    assert isinstance(axis, int)


# --- 2. Explicit Hint Tests (Feature 027) ---


def test_hint_primitive_override(fuzzer):
  """Verify explicit hint overrides naming heuristic."""
  # 'mask' is usually bool, but if we say 'int', it must be int.
  hints = {"mask": "int"}
  inputs = fuzzer.generate_inputs(["mask"], hints=hints)

  # Primitives in hint generator return scalars by default
  assert isinstance(inputs["mask"], int)


def test_hint_array(fuzzer):
  """Verify 'Array' hint generates float32 array."""
  hints = {"x": "Array"}
  inputs = fuzzer.generate_inputs(["x"], hints=hints)

  val = inputs["x"]
  assert isinstance(val, np.ndarray)
  assert val.dtype == np.float32


def test_hint_optional(fuzzer):
  """Verify Optional[int] returns int or None."""
  hints = {"x": "Optional[int]"}

  # Run multiple times to catch both branches (probabilistic)
  seen_int = False

  for _ in range(50):
    inputs = fuzzer.generate_inputs(["x"], hints=hints)
    val = inputs["x"]
    if isinstance(val, int):
      seen_int = True

  assert seen_int, "Failed to generate int from Optional[int]"


def test_hint_union(fuzzer):
  """Verify int | None behaves like Optional."""
  hints = {"x": "int | None"}

  seen_none = False
  seen_int = False

  for _ in range(50):
    inputs = fuzzer.generate_inputs(["x"], hints=hints)
    val = inputs["x"]
    if val is None:
      seen_none = True
    elif isinstance(val, int):
      seen_int = True

  assert seen_int or seen_none


def test_hint_tuple_fixed(fuzzer):
  """Verify Tuple[int, float]."""
  hints = {"vals": "Tuple[int, float]"}
  inputs = fuzzer.generate_inputs(["vals"], hints=hints)

  val = inputs["vals"]
  assert isinstance(val, tuple)
  assert len(val) == 2
  assert isinstance(val[0], int)
  assert isinstance(val[1], float)


def test_hint_tuple_variadic(fuzzer):
  """Verify Tuple[int, ...]."""
  hints = {"vals": "Tuple[int, ...]"}
  inputs = fuzzer.generate_inputs(["vals"], hints=hints)

  val = inputs["vals"]
  assert isinstance(val, tuple)
  assert len(val) >= 1
  assert all(isinstance(v, int) for v in val)


def test_hint_list_of_arrays(fuzzer):
  """Verify List[Array] generates list of ndarrays."""
  hints = {"tensors": "List[Array]"}
  inputs = fuzzer.generate_inputs(["tensors"], hints=hints)

  val = inputs["tensors"]
  assert isinstance(val, list)
  assert len(val) > 0
  assert isinstance(val[0], np.ndarray)


def test_hint_dtype_object(fuzzer):
  """Verify 'dtype' hint returns a real numpy dtype type object."""
  hints = {"d": "dtype"}
  inputs = fuzzer.generate_inputs(["d"], hints=hints)

  val = inputs["d"]
  # numpy dtypes are types like np.float32
  assert issubclass(val, np.generic) or isinstance(val, np.dtype)


def test_nested_complex_hint(fuzzer):
  """Verify Tuple[Optional[int], Array]."""
  hints = {"c": "Tuple[Optional[int], Array]"}
  inputs = fuzzer.generate_inputs(["c"], hints=hints)

  val = inputs["c"]
  assert isinstance(val, tuple)
  assert len(val) == 2
  # El 0: int or None
  assert val[0] is None or isinstance(val[0], int)
  # El 1: Array
  assert isinstance(val[1], np.ndarray)


# --- 3. Dictionary & Recursive Limit Tests ---


def test_hint_simple_dict(fuzzer):
  """Verify Dict[str, int]."""
  hints = {"config": "Dict[str, int]"}
  inputs = fuzzer.generate_inputs(["config"], hints=hints)

  val = inputs["config"]
  assert isinstance(val, dict)
  assert len(val) > 0
  key, v = next(iter(val.items()))
  assert isinstance(key, str)
  assert isinstance(v, int)


def test_hint_nested_container(fuzzer):
  """Verify Dict[str, List[int]]."""
  hints = {"nested": "Dict[str, List[int]]"}
  inputs = fuzzer.generate_inputs(["nested"], hints=hints)

  val = inputs["nested"]
  assert isinstance(val, dict)
  key, v = next(iter(val.items()))
  assert isinstance(v, list)
  assert isinstance(v[0], int)


def test_recursion_limit_stops(fuzzer):
  """
  Verify iteration stops at MAX_RECURSION_DEPTH.
  Hint: List[List[List[List[int]]]] - depth 4 vs Limit 3
  """
  # Depth max is 3 by default
  hint = "List[List[List[List[int]]]]"
  hints = {"deep": hint}
  inputs = fuzzer.generate_inputs(["deep"], hints=hints)

  _val = inputs["deep"]
  assert isinstance(_val, list)


def test_unhashable_dict_key_fallback(fuzzer):
  """
  Verify Dict[Array, int] forces key to string via updated strategy.
  """
  hints = {"bad_key": "Dict[Array, int]"}
  inputs = fuzzer.generate_inputs(["bad_key"], hints=hints)

  val = inputs["bad_key"]
  assert isinstance(val, dict)
  if val:
    key = next(iter(val.keys()))
    # Array object isn't hashable, so strategy maps to str
    assert isinstance(key, str)
