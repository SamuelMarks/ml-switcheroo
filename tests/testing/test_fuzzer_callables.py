"""
Tests for Fuzzer Callable Support.

Verifies that:
1. Hints like 'Callable' generate executable functions.
2. Generated functions act as identity or simple stubs.
3. Logic parses 'Callable[[int], int]' correctly.
4. 'func' and 'function' shorthands are supported.
"""

import pytest
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck
from typing import Callable

from ml_switcheroo.testing.fuzzer.core import InputFuzzer


@pytest.fixture
def fuzzer():
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_simple_callable(fuzzer, data):
  """
  Scenario: User hints 'Callable'.
  Expectation: Return a python function (lambda).
  """
  hints = {"fn": "Callable"}
  strats = fuzzer.build_strategies(["fn"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))

  val = inputs["fn"]
  assert callable(val)

  # Verify behavior: identity
  assert val(5) == 5
  assert val("foo") == "foo"


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_complex_callable_hint(fuzzer, data):
  """
  Scenario: User hints 'Callable[[int], int]'.
  Expectation: Fuzzer handles the brackets gracefully and returns basic callable.
  """
  hints = {"op": "Callable[[int], int]"}
  strats = fuzzer.build_strategies(["op"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))

  val = inputs["op"]
  assert callable(val)
  assert val(10) == 10


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_func_shorthand(fuzzer, data):
  """
  Scenario: User hints 'func' or 'function' (often used in ODL for vmap/grad).
  """
  hints = {"f": "func", "g": "function"}
  strats = fuzzer.build_strategies(["f", "g"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))

  assert callable(inputs["f"])
  assert callable(inputs["g"])

  # Check multi-arg support (for vmap simulation)
  # lambda x, *args: x
  assert inputs["f"](1, 2, 3) == 1


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_callable_in_list(fuzzer, data):
  """
  Scenario: List[Callable].
  Expectation: A list of functions.
  """
  hints = {"ops": "List[Callable]"}
  strats = fuzzer.build_strategies(["ops"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))

  lst = inputs["ops"]
  assert isinstance(lst, list)
  if lst:
    assert callable(lst[0])


def test_fallback_depth_recursion(fuzzer):
  """
  Scenario: Recursion limit hit on 'List[List[List[Callable]]]'.
  Expectation: Should still return sensible fallbacks for containers,
               or at least not crash if it reaches the callable.
  """
  # Force minimal depth to trigger fallback logic logic immediately
  # We test `get_fallback_base_value` logic directly for precision
  from ml_switcheroo.testing.fuzzer.parser import get_fallback_base_value

  val = get_fallback_base_value("Callable", (1, 1))
  assert callable(val)
  assert val("test") == "test"


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_vmap_usage_simulation(fuzzer, data):
  """
  Scenario: Simulating a vmap test where 'func' is generated.
  """
  hints = {"func": "Callable", "in_axes": "int"}
  strats = fuzzer.build_strategies(["func", "in_axes"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))

  fn = inputs["func"]

  # Simulate JAX usage: output = fn(input)
  arr = np.array([1, 2, 3])
  out = fn(arr)

  assert np.array_equal(arr, out)
