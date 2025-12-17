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
from typing import List, Dict

from ml_switcheroo.testing.fuzzer import InputFuzzer
from ml_switcheroo.frameworks import register_framework, get_adapter


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
  """Verify scalar names generate Python scalars via heuristics."""
  # alpha, eps -> floats
  inputs = fuzzer.generate_inputs(params=["alpha", "eps"])
  assert isinstance(inputs["alpha"], float)
  assert isinstance(inputs["eps"], float)


def test_axis_heuristic_validity(fuzzer):
  """Verify 'axis' is within bounds of the generated shape."""
  # We rely on internal _get_random_shape consistency for independent calls
  # but generate_inputs generates one shape per call context for heuristics.
  for _ in range(10):
    # We need another param 'x' so shape is generated/used implicitly
    inputs = fuzzer.generate_inputs(["x", "axis"])
    x = inputs["x"]
    axis = inputs["axis"]

    assert isinstance(x, np.ndarray)
    rank = len(x.shape)
    assert isinstance(axis, int)
    assert 0 <= axis < rank


# --- 2. Explicit Hint Tests ---


def test_hint_primitive_override(fuzzer):
  """Verify explicit hint overrides naming heuristic."""
  # 'mask' is usually bool array, but if we say 'int', it must be int scalar.
  hints = {"mask": "int"}
  inputs = fuzzer.generate_inputs(["mask"], hints=hints)
  assert isinstance(inputs["mask"], int)


def test_hint_array(fuzzer):
  """Verify 'Array' hint generates float32 array."""
  hints = {"x": "Array"}
  inputs = fuzzer.generate_inputs(["x"], hints=hints)
  val = inputs["x"]
  assert isinstance(val, np.ndarray)
  assert val.dtype == np.float32


def test_hint_union(fuzzer):
  """Verify int | float behaves like Union."""
  hints = {"x": "int | float"}
  seen_types = set()
  for _ in range(50):
    val = fuzzer.generate_inputs(["x"], hints=hints)["x"]
    seen_types.add(type(val))

  assert int in seen_types
  assert float in seen_types


def test_hint_tuple_variadic(fuzzer):
  """Verify Tuple[int, ...]."""
  hints = {"vals": "Tuple[int, ...]"}
  inputs = fuzzer.generate_inputs(["vals"], hints=hints)
  val = inputs["vals"]
  assert isinstance(val, tuple)
  assert len(val) >= 1
  assert all(isinstance(v, int) for v in val)


def test_hint_nested_complex(fuzzer):
  """Verify Dict[str, List[int]]."""
  hints = {"config": "Dict[str, List[int]]"}
  inputs = fuzzer.generate_inputs(["config"], hints=hints)
  val = inputs["config"]
  assert isinstance(val, dict)
  if val:
    k, v = next(iter(val.items()))
    assert isinstance(k, str)
    assert isinstance(v, list)
    if v:
      assert isinstance(v[0], int)


def test_recursion_limit_stops(fuzzer):
  """Verify iteration stops at MAX_RECURSION_DEPTH."""
  # Depth max is 3 by default
  hint = "List[List[List[List[int]]]]"
  hints = {"deep": hint}
  inputs = fuzzer.generate_inputs(["deep"], hints=hints)
  _val = inputs["deep"]
  # Should return valid object (likely empty list due to fallback)
  assert isinstance(_val, list)


def test_unhashable_dict_key_fallback(fuzzer):
  """Verify Dict[Array, int] converts key to string (safe fallback)."""
  hints = {"bad_key": "Dict[Array, int]"}
  inputs = fuzzer.generate_inputs(["bad_key"], hints=hints)
  val = inputs["bad_key"]
  if val:
    key = next(iter(val.keys()))
    assert isinstance(key, str)


def test_dtype_object_generation(fuzzer):
  """Verify 'dtype' hint returns a real numpy dtype."""
  hints = {"d": "dtype"}
  inputs = fuzzer.generate_inputs(["d"], hints=hints)
  val = inputs["d"]
  # numpy dtypes are types like np.float32 or instance np.dtype('float32')
  assert isinstance(val, type) or isinstance(val, np.dtype)


# --- 3. Symbolic Constraints Tests (The Core Objective) ---


def test_symbolic_sharing(fuzzer):
  """
  Scenario: Two inputs `x` and `y` share a dimension `N`.
  Hint: x: Array['N'], y: Array['N']
  Expect: x.shape == y.shape
  """
  hints = {"x": "Array['N']", "y": "Array['N']"}
  inputs = fuzzer.generate_inputs(["x", "y"], hints=hints)

  x = inputs["x"]
  y = inputs["y"]

  assert isinstance(x, np.ndarray)
  assert isinstance(y, np.ndarray)
  assert x.shape == y.shape
  assert len(x.shape) == 1

  # Ensure dimensions aren't always 0 or 1 (trivial pass)
  # The fuzzer defaults to rand(2, 6) for symbols
  assert x.shape[0] >= 2


def test_matmul_constraints(fuzzer):
  """
  Scenario: Matmul (A, B) @ (B, C) -> (A, C).
  Hint: x: Array['A', 'B'], y: Array['B', 'C']
  Expect: x.shape[1] == y.shape[0]
  """
  hints = {"x": "Array['A', 'B']", "y": "Array['B', 'C']"}
  inputs = fuzzer.generate_inputs(["x", "y"], hints=hints)

  x = inputs["x"]
  y = inputs["y"]

  assert len(x.shape) == 2
  assert len(y.shape) == 2
  # Inner dimension must match
  assert x.shape[1] == y.shape[0]


def test_fixed_dimension_mixed(fuzzer):
  """
  Scenario: Fixed dimension mixed with symbolic.
  Hint: x: Array[3, 'D']
  Expect: shape[0] == 3.
  """
  hints = {"x": "Array[3, 'D']"}
  inputs = fuzzer.generate_inputs(["x"], hints=hints)

  x = inputs["x"]
  assert x.shape[0] == 3
  assert len(x.shape) == 2


def test_symbolic_list_consistency(fuzzer):
  """
  Scenario: List of arrays sharing a symbol.
  Hint: x: List[Array['Z']]
  Expect: All arrays in list have same shape (Z,).
  """
  hints = {"x": "List[Array['Z']]"}
  inputs = fuzzer.generate_inputs(["x"], hints=hints)

  lst = inputs["x"]
  if not lst:
    return  # Valid if list empty

  shape0 = lst[0].shape
  for arr in lst[1:]:
    assert arr.shape == shape0


def test_independent_calls_are_independent(fuzzer):
  """
  Scenario: Two separate calls to generate_inputs regarding 'N'.
  Expect: 'N' can be different between calls (context is scoped).
  """
  hints = {"x": "Array['N']"}

  res1 = fuzzer.generate_inputs(["x"], hints=hints)
  s1 = res1["x"].shape[0]

  # Force a seed reset or just run many times to get difference?
  # Fuzzer uses random, so it should vary.
  diff_found = False
  for i in range(20):
    res2 = fuzzer.generate_inputs(["x"], hints=hints)
    s2 = res2["x"].shape[0]
    if s1 != s2:
      diff_found = True
      break

  assert diff_found, "Symbolic resolution across calls should be independent/random."


# --- 4. Framework Adaptation Tests ---


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
