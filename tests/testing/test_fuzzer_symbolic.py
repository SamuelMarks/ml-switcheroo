"""
Tests for Robust Fuzzer Dtype Support, Hint Parsing, and Dynamic Shapes.

Verifies:
1. Heuristics fallback (legacy behavior).
2. Explicit Hint Parsing (Feature 027 integration).
3. Symbolic Shape Constraints (e.g. Array['N']).
4. Complex Nested Types (List[int], Tuple[int, ...], Dict[str, int]).
5. Recursion limits on deep nesting.
"""

import pytest
import numpy as np
import random
import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck
from typing import List, Dict

from ml_switcheroo.testing.fuzzer import InputFuzzer
from ml_switcheroo.frameworks import register_framework, get_adapter


@pytest.fixture
def fuzzer():
  # Fix seed for predictable generation
  random.seed(42)
  np.random.seed(42)
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_heuristic_booleans(fuzzer, data):
  """Verify 'mask' generates boolean array."""
  strats = fuzzer.build_strategies(["mask", "condition"])
  inputs = data.draw(st.fixed_dictionaries(strats))
  assert inputs["mask"].dtype == bool
  assert inputs["condition"].dtype == bool


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_heuristic_integers(fuzzer, data):
  """Verify 'indices' generates int array."""
  strats = fuzzer.build_strategies(["indices", "k_idx"])
  inputs = data.draw(st.fixed_dictionaries(strats))
  assert np.issubdtype(inputs["indices"].dtype, np.integer)
  assert np.issubdtype(inputs["k_idx"].dtype, np.integer)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_heuristic_scalars(fuzzer, data):
  """Verify scalar names generate Python scalars via heuristics."""
  # alpha, eps -> floats
  strats = fuzzer.build_strategies(["alpha", "eps"])
  inputs = data.draw(st.fixed_dictionaries(strats))
  assert isinstance(inputs["alpha"], float)
  assert isinstance(inputs["eps"], float)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_axis_heuristic_validity(fuzzer, data):
  """Verify 'axis' heuristic generates integer type."""
  strats = fuzzer.build_strategies(["x", "axis"])
  inputs = data.draw(st.fixed_dictionaries(strats))
  x = inputs["x"]
  axis = inputs["axis"]

  assert isinstance(x, np.ndarray)
  assert isinstance(axis, int)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hint_primitive_override(fuzzer, data):
  """Verify explicit hint overrides naming heuristic."""
  hints = {"mask": "int"}
  strats = fuzzer.build_strategies(["mask"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  assert isinstance(inputs["mask"], int)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hint_array(fuzzer, data):
  """Verify 'Array' hint generates float32 array."""
  hints = {"x": "Array"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["x"]
  assert isinstance(val, np.ndarray)
  assert val.dtype == np.float32


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hint_union(fuzzer, data):
  """Verify int | float behaves like Union."""
  hints = {"x": "int | float"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  val = data.draw(st.fixed_dictionaries(strats))["x"]
  assert isinstance(val, (int, float))


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hint_tuple_variadic(fuzzer, data):
  """Verify Tuple[int, ...]."""
  hints = {"vals": "Tuple[int, ...]"}
  strats = fuzzer.build_strategies(["vals"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["vals"]
  assert isinstance(val, tuple)
  assert len(val) >= 1
  assert all(isinstance(v, int) for v in val)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hint_nested_complex(fuzzer, data):
  """Verify Dict[str, List[int]]."""
  hints = {"config": "Dict[str, List[int]]"}
  strats = fuzzer.build_strategies(["config"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["config"]
  assert isinstance(val, dict)
  if val:
    k, v = next(iter(val.items()))
    assert isinstance(k, str)
    assert isinstance(v, list)
    if v:
      assert isinstance(v[0], int)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_recursion_limit_stops(fuzzer, data):
  """Verify iteration stops at appropriate depth."""
  hint = "List[List[List[List[int]]]]"
  hints = {"deep": hint}
  strats = fuzzer.build_strategies(["deep"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  _val = inputs["deep"]
  assert isinstance(_val, list)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_unhashable_dict_key_fallback(fuzzer, data):
  """Verify Dict[Array, int] converts key to string (safe fallback)."""
  hints = {"bad_key": "Dict[Array, int]"}
  strats = fuzzer.build_strategies(["bad_key"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["bad_key"]
  if val:
    key = next(iter(val.keys()))
    assert isinstance(key, str)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_dtype_object_generation(fuzzer, data):
  """Verify 'dtype' hint returns a real numpy dtype."""
  hints = {"d": "dtype"}
  strats = fuzzer.build_strategies(["d"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["d"]
  assert isinstance(val, type) or isinstance(val, np.dtype)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_symbolic_sharing(fuzzer, data):
  """
  Scenario: Two inputs `x` and `y` share a dimension `N`.
  Hint: x: Array['N'], y: Array['N']
  Expect: x.shape == y.shape
  """
  hints = {"x": "Array['N']", "y": "Array['N']"}
  strats = fuzzer.build_strategies(["x", "y"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))

  x = inputs["x"]
  y = inputs["y"]

  assert isinstance(x, np.ndarray)
  assert isinstance(y, np.ndarray)
  assert x.shape == y.shape
  assert len(x.shape) == 1

  assert x.shape[0] >= 1


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_matmul_constraints(fuzzer, data):
  """
  Scenario: Matmul (A, B) @ (B, C) -> (A, C).
  Hint: x: Array['A', 'B'], y: Array['B', 'C']
  Expect: x.shape[1] == y.shape[0]
  """
  hints = {"x": "Array['A', 'B']", "y": "Array['B', 'C']"}
  strats = fuzzer.build_strategies(["x", "y"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))

  x = inputs["x"]
  y = inputs["y"]

  assert len(x.shape) == 2
  assert len(y.shape) == 2
  # Inner dimension must match
  assert x.shape[1] == y.shape[0]


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fixed_dimension_mixed(fuzzer, data):
  """
  Scenario: Fixed dimension mixed with symbolic.
  Hint: x: Array[3, 'D']
  Expect: shape[0] == 3.
  """
  hints = {"x": "Array[3, 'D']"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  x = inputs["x"]
  assert x.shape[0] == 3
  assert len(x.shape) == 2


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_symbolic_list_consistency(fuzzer, data):
  """
  Scenario: List of arrays sharing a symbol.
  Hint: x: List[Array['Z']]
  Expect: All arrays in list have same shape (Z,).
  """
  hints = {"x": "List[Array['Z']]"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))

  lst = inputs["x"]
  if not lst:
    return  # Valid if list empty

  shape0 = lst[0].shape
  for arr in lst[1:]:
    assert arr.shape == shape0


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_independent_calls_are_independent(fuzzer, data):
  """
  Scenario: Two separate calls to generate_inputs regarding 'N'.
  Expect: 'N' can be different between calls (context is scoped).
  """
  hints = {"x": "Array['N']"}

  strats1 = fuzzer.build_strategies(["x"], hints=hints)
  res1 = data.draw(st.fixed_dictionaries(strats1))

  strats2 = fuzzer.build_strategies(["x"], hints=hints)
  res2 = data.draw(st.fixed_dictionaries(strats2))

  assert isinstance(res1["x"], np.ndarray)
  assert isinstance(res2["x"], np.ndarray)


def test_adapt_to_framework_passthrough(fuzzer):
  """Verify that if adapter is missing or fails, it returns raw data."""
  raw = {"x": np.array([1])}

  # "unknown_fw" has no adapter -> passthrough in fuzzer
  res = fuzzer.adapt_to_framework(raw, "unknown_fw")
  assert res is raw


def test_adapt_to_framework_delegation(fuzzer):
  """Verify fuzzer calls adapter.convert."""

  # Register a dummy adapter
  @register_framework("mock_fw")
  class MockAdapter:
    def convert(self, x):
      return "converted"

  raw = {"x": 1}
  res = fuzzer.adapt_to_framework(raw, "mock_fw")
  assert res["x"] == "converted"
