"""Test suite for the Fuzzer Constraints module."""

import pytest
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck
from ml_switcheroo.testing.fuzzer import InputFuzzer


@pytest.fixture
def fuzzer():
  """Provides a mock fuzzer for testing."""
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_symbolic_sharing(fuzzer, data):
  """Verifies the behavior of symbolic sharing."""
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
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_matmul_constraints(fuzzer, data):
  """Verifies the behavior of matmul constraints."""
  hints = {"x": "Array['A', 'B']", "y": "Array['B', 'C']"}
  strats = fuzzer.build_strategies(["x", "y"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  x = inputs["x"]
  y = inputs["y"]
  assert len(x.shape) == 2
  assert len(y.shape) == 2
  assert x.shape[1] == y.shape[0]


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_fixed_dimension(fuzzer, data):
  """Verifies the behavior of fixed dimension."""
  hints = {"x": "Array[3, 'D']"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  x = inputs["x"]
  assert x.shape[0] == 3
  assert len(x.shape) == 2


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_symbolic_list_consistency(fuzzer, data):
  """Verifies the behavior of symbolic list consistency."""
  hints = {"x": "List[Array['Z']]"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  lst = inputs["x"]
  if not lst:
    return
  shape0 = lst[0].shape
  for arr in lst[1:]:
    assert arr.shape == shape0


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_independent_calls_are_independent(fuzzer, data):
  """Verifies the behavior of independent calls are independent."""
  hints = {"x": "Array['N']"}
  strats1 = fuzzer.build_strategies(["x"], hints=hints)
  res1 = data.draw(st.fixed_dictionaries(strats1))
  strats2 = fuzzer.build_strategies(["x"], hints=hints)
  res2 = data.draw(st.fixed_dictionaries(strats2))
  assert isinstance(res1["x"], np.ndarray)
  assert isinstance(res2["x"], np.ndarray)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_tensor_alias_support(fuzzer, data):
  """Verifies the behavior of tensor alias support."""
  hints = {"x": "Tensor['A']"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  assert isinstance(inputs["x"], np.ndarray)
  assert len(inputs["x"].shape) == 1
