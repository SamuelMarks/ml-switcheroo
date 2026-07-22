"""Test suite for the Fuzzer Callables module."""

import pytest
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck
from ml_switcheroo.testing.fuzzer.core import InputFuzzer


@pytest.fixture
def fuzzer():
  """Provides a mock fuzzer for testing."""
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_simple_callable(fuzzer, data):
  """Generates simple callable."""
  hints = {"fn": "Callable"}
  strats = fuzzer.build_strategies(["fn"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["fn"]
  assert callable(val)
  assert val(5) == 5
  assert val("foo") == "foo"


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_complex_callable_hint(fuzzer, data):
  """Generates complex callable hint."""
  hints = {"op": "Callable[[int], int]"}
  strats = fuzzer.build_strategies(["op"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["op"]
  assert callable(val)
  assert val(10) == 10


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_func_shorthand(fuzzer, data):
  """Generates function shorthand."""
  hints = {"f": "func", "g": "function"}
  strats = fuzzer.build_strategies(["f", "g"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  assert callable(inputs["f"])
  assert callable(inputs["g"])
  assert inputs["f"](1, 2, 3) == 1


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_callable_in_list(fuzzer, data):
  """Verifies the behavior of callable in list."""
  hints = {"ops": "List[Callable]"}
  strats = fuzzer.build_strategies(["ops"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  lst = inputs["ops"]
  assert isinstance(lst, list)
  if lst:
    assert callable(lst[0])


def test_fallback_depth_recursion(fuzzer):
  """Verifies the behavior of fallback depth recursion."""
  from ml_switcheroo.testing.fuzzer.parser import get_fallback_base_value

  val = get_fallback_base_value("Callable", (1, 1))
  assert callable(val)
  assert val("test") == "test"


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_vmap_usage_simulation(fuzzer, data):
  """Verifies the behavior of vmap usage simulation."""
  hints = {"func": "Callable", "in_axes": "int"}
  strats = fuzzer.build_strategies(["func", "in_axes"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  fn = inputs["func"]
  arr = np.array([1, 2, 3])
  out = fn(arr)
  assert np.array_equal(arr, out)
