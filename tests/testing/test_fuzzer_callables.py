"""Test suite for the Fuzzer Callables module."""

from typing import Any, Callable, Dict, List

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from ml_switcheroo.testing.fuzzer.core import InputFuzzer


@pytest.fixture
def fuzzer() -> InputFuzzer:
  """Docstring."""
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_simple_callable(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Generates simple callable."""
  hints: Dict[str, str] = {"fn": "Callable"}
  strats: Dict[str, st.SearchStrategy[Any]] = fuzzer.build_strategies(["fn"], hints=hints)
  inputs: Dict[str, Any] = data.draw(st.fixed_dictionaries(strats))
  val: Callable = inputs["fn"]
  assert callable(val)
  assert val(5) == 5
  assert val("foo") == "foo"


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_complex_callable_hint(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Generates complex callable hint."""
  hints: Dict[str, str] = {"op": "Callable[[int], int]"}
  strats: Dict[str, st.SearchStrategy[Any]] = fuzzer.build_strategies(["op"], hints=hints)
  inputs: Dict[str, Any] = data.draw(st.fixed_dictionaries(strats))
  val: Callable = inputs["op"]
  assert callable(val)
  assert val(10) == 10


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generate_func_shorthand(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Generates function shorthand."""
  hints: Dict[str, str] = {"f": "func", "g": "function"}
  strats: Dict[str, st.SearchStrategy[Any]] = fuzzer.build_strategies(["f", "g"], hints=hints)
  inputs: Dict[str, Any] = data.draw(st.fixed_dictionaries(strats))
  assert callable(inputs["f"])
  assert callable(inputs["g"])
  assert inputs["f"](1, 2, 3) == 1


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_callable_in_list(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of callable in list."""
  hints: Dict[str, str] = {"ops": "List[Callable]"}
  strats: Dict[str, st.SearchStrategy[Any]] = fuzzer.build_strategies(["ops"], hints=hints)
  inputs: Dict[str, Any] = data.draw(st.fixed_dictionaries(strats))
  lst: List[Any] = inputs["ops"]
  assert isinstance(lst, list)
  if lst:
    assert callable(lst[0])


def test_fallback_depth_recursion(fuzzer: InputFuzzer) -> None:
  """Verifies the behavior of fallback depth recursion."""
  from ml_switcheroo.testing.fuzzer.parser import get_fallback_base_value
  from ml_switcheroo.testing.fuzzer.type_parser import parse_type_annotation

  val: Any = get_fallback_base_value(parse_type_annotation("Callable"), (1, 1))
  assert callable(val)
  assert val("test") == "test"


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_vmap_usage_simulation(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of vmap usage simulation."""
  hints: Dict[str, str] = {"func": "Callable", "in_axes": "int"}
  strats: Dict[str, st.SearchStrategy[Any]] = fuzzer.build_strategies(["func", "in_axes"], hints=hints)
  inputs: Dict[str, Any] = data.draw(st.fixed_dictionaries(strats))
  fn: Callable = inputs["func"]
  arr: np.ndarray = np.array([1, 2, 3])
  out: Any = fn(arr)
  assert np.array_equal(arr, out)
