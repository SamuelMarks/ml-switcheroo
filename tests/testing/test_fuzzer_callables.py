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
from typing import Callable

from ml_switcheroo.testing.fuzzer.core import InputFuzzer


@pytest.fixture
def fuzzer():
  return InputFuzzer()


def test_generate_simple_callable(fuzzer):
  """
  Scenario: User hints 'Callable'.
  Expectation: Return a python function (lambda).
  """
  hints = {"fn": "Callable"}
  inputs = fuzzer.generate_inputs(["fn"], hints=hints)

  val = inputs["fn"]
  assert callable(val)

  # Verify behavior: identity
  assert val(5) == 5
  assert val("foo") == "foo"


def test_generate_complex_callable_hint(fuzzer):
  """
  Scenario: User hints 'Callable[[int], int]'.
  Expectation: Fuzzer handles the brackets gracefully and returns basic callable.
  """
  hints = {"op": "Callable[[int], int]"}
  inputs = fuzzer.generate_inputs(["op"], hints=hints)

  val = inputs["op"]
  assert callable(val)
  assert val(10) == 10


def test_generate_func_shorthand(fuzzer):
  """
  Scenario: User hints 'func' or 'function' (often used in ODL for vmap/grad).
  """
  hints = {"f": "func", "g": "function"}
  inputs = fuzzer.generate_inputs(["f", "g"], hints=hints)

  assert callable(inputs["f"])
  assert callable(inputs["g"])

  # Check multi-arg support (for vmap simulation)
  # lambda x, *args: x
  assert inputs["f"](1, 2, 3) == 1


def test_callable_in_list(fuzzer):
  """
  Scenario: List[Callable].
  Expectation: A list of functions.
  """
  hints = {"ops": "List[Callable]"}
  inputs = fuzzer.generate_inputs(["ops"], hints=hints)

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


def test_vmap_usage_simulation(fuzzer):
  """
  Scenario: Simulating a vmap test where 'func' is generated.
  """
  hints = {"func": "Callable", "in_axes": "int"}
  inputs = fuzzer.generate_inputs(["func", "in_axes"], hints=hints)

  fn = inputs["func"]

  # Simulate JAX usage: output = fn(input)
  arr = np.array([1, 2, 3])
  out = fn(arr)

  assert np.array_equal(arr, out)
