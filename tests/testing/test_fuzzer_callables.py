"""Test suite for the Fuzzer Callables module."""

import typing

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from ml_switcheroo.testing.fuzzer.core import InputFuzzer


@pytest.fixture
def fuzzer() -> InputFuzzer:
  """Execute fuzzer helper logic.

  Returns:
      Computed test result or fixture value.
  """
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_simple_callable(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Generates simple callable.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: typing.Dict[str, str] = {"fn": "Callable"}
  strats = fuzzer.build_strategies(["fn"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["fn"]
  assert callable(val)
  assert val(5) == 5  # type: ignore[unreachable]
  assert val("foo") == "foo"


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_complex_callable_hint(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Generates complex callable hint.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: typing.Dict[str, str] = {"op": "Callable[[int], int]"}
  strats = fuzzer.build_strategies(["op"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["op"]
  assert callable(val)
  assert val(10) == 10  # type: ignore[unreachable]


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_func_shorthand(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Generates function shorthand.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: typing.Dict[str, str] = {"f": "func", "g": "function"}
  strats = fuzzer.build_strategies(["f", "g"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  assert callable(inputs["f"])
  assert callable(inputs["g"])  # type: ignore[unreachable]
  assert inputs["f"](1, 2, 3) == 1


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_callable_in_list(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of callable in list.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: typing.Dict[str, str] = {"ops": "List[Callable]"}
  strats = fuzzer.build_strategies(["ops"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  lst = inputs["ops"]
  assert isinstance(lst, list)
  if lst:
    assert callable(lst[0])


def test_fallback_depth_recursion(fuzzer: InputFuzzer) -> None:
  """Verifies the behavior of fallback depth recursion.

  Args:
      fuzzer: Test argument or fixture.
  """
  from ml_switcheroo.testing.fuzzer.parser import get_fallback_base_value
  from ml_switcheroo.testing.fuzzer.type_parser import parse_type_annotation

  val = get_fallback_base_value(parse_type_annotation("Callable"), (1, 1))
  assert callable(val)
  assert val("test") == "test"  # type: ignore[unreachable]


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_vmap_usage_simulation(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of vmap usage simulation.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: typing.Dict[str, str] = {"func": "Callable", "in_axes": "int"}
  strats = fuzzer.build_strategies(["func", "in_axes"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  fn = inputs["func"]
  arr = np.array([1, 2, 3])
  out = fn(arr)  # type: ignore[operator]
  assert np.array_equal(arr, out)
