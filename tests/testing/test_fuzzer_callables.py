"""Test suite for the Fuzzer Callables module."""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from ml_switcheroo.testing.fuzzer.core import InputFuzzer


@pytest.fixture
def fuzzer() -> InputFuzzer:
  """Create a configured `InputFuzzer` fixture for fuzzer callables tests.

  Returns:
      Configured `fuzzer` fixture value (InputFuzzer).
  """
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
def test_generate_simple_callable(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Generates simple callable.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"fn": "Callable"}
  strats = fuzzer.build_strategies(["fn"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["fn"]
  assert callable(val)
  assert val(5) == 5
  assert val("foo") == "foo"


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
def test_generate_complex_callable_hint(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Generates complex callable hint.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"op": "Callable[[int], int]"}
  strats = fuzzer.build_strategies(["op"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["op"]
  assert callable(val)
  assert val(10) == 10


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
def test_generate_func_shorthand(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Generates function shorthand.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"f": "func", "g": "function"}
  strats = fuzzer.build_strategies(["f", "g"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  assert callable(inputs["f"])
  assert callable(inputs["g"])
  assert inputs["f"](1, 2, 3) == 1


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
def test_callable_in_list(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles callable in list as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"ops": "List[Callable]"}
  strats = fuzzer.build_strategies(["ops"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  lst = inputs["ops"]
  assert isinstance(lst, list)
  assert len(lst) >= 1
  assert len(lst) <= 4
  assert callable(lst[0])
  fn0 = lst[0]
  assert callable(fn0)
  assert fn0(42) == 42
  assert fn0("hello") == "hello"


def test_fallback_depth_recursion(fuzzer: InputFuzzer) -> None:
  """Verify `callable` handles fallback depth recursion as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
  """
  from ml_switcheroo.testing.fuzzer.parser import get_fallback_base_value
  from ml_switcheroo.testing.fuzzer.type_parser import parse_type_annotation

  val = get_fallback_base_value(parse_type_annotation("Callable"), (1, 1))
  assert callable(val)
  assert val("test") == "test"


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
def test_vmap_usage_simulation(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles vmap usage simulation as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"func": "Callable", "in_axes": "int"}
  strats = fuzzer.build_strategies(["func", "in_axes"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  fn = inputs["func"]
  assert callable(fn)
  arr = np.array([1, 2, 3])
  out = fn(arr)
  assert isinstance(out, np.ndarray)
  assert np.array_equal(arr, out)
