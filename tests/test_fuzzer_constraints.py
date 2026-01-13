"""
Tests for Symbolic Shape Constraints in InputFuzzer.

Verifies that:
1. `Array['B', 'N']` generates consistent shapes across arguments.
2. Fixed integers in hints (`Array[32, 'N']`) are respected.
3. Symbols are resolved consistently within a single `build_strategies` call.
"""

import pytest
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck
from ml_switcheroo.testing.fuzzer import InputFuzzer


@pytest.fixture
def fuzzer():
  return InputFuzzer()


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
def test_fixed_dimension(fuzzer, data):
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
  # Fuzzer generates list length 1-3.
  if not lst:
    return  # Valid empty list

  shape0 = lst[0].shape
  for arr in lst[1:]:
    assert arr.shape == shape0


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_independent_calls_are_independent(fuzzer, data):
  """
  Scenario: Two separate calls to generate regarding 'N'.
  Expect: 'N' can be different between calls.
  """
  hints = {"x": "Array['N']"}

  # Call 1
  strats1 = fuzzer.build_strategies(["x"], hints=hints)
  res1 = data.draw(st.fixed_dictionaries(strats1))

  # Call 2 (rebuild strategy to reset symbol map context if it was internal, but build_strategies creates new scope)
  strats2 = fuzzer.build_strategies(["x"], hints=hints)
  res2 = data.draw(st.fixed_dictionaries(strats2))

  assert isinstance(res1["x"], np.ndarray)
  assert isinstance(res2["x"], np.ndarray)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_tensor_alias_support(fuzzer, data):
  """
  Verify 'Tensor' keyword works same as 'Array'.
  """
  hints = {"x": "Tensor['A']"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))

  assert isinstance(inputs["x"], np.ndarray)
  assert len(inputs["x"].shape) == 1
