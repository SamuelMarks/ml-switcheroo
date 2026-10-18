"""Test suite for the Dsl Dtype module."""

import typing

from typing import Any, Dict

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from ml_switcheroo.core.dsl import ParameterDef
from ml_switcheroo.testing.fuzzer import InputFuzzer


def test_param_dtype_field_initialization() -> None:
  """Verifies the behavior of parameter dtype field initialization."""
  p: ParameterDef = ParameterDef(name="idx", dtype="int64")
  assert getattr(p, "dtype") == "int64"
  assert getattr(p, "name") == "idx"


def test_param_dtype_default_is_none() -> None:
  """Verifies the behavior of parameter dtype default is none."""
  p: ParameterDef = ParameterDef(name="x")
  assert getattr(p, "dtype") is None


def test_param_dtype_valid_types() -> None:
  """Verifies the behavior of parameter dtype valid types."""
  p1: ParameterDef = ParameterDef(name="mask", dtype="bool")
  assert getattr(p1, "dtype") == "bool"
  p2: ParameterDef = ParameterDef(name="embedding", dtype="float16")
  assert getattr(p2, "dtype") == "float16"


def test_param_dtype_and_rank() -> None:
  """Verifies the behavior of parameter dtype and rank."""
  p: ParameterDef = ParameterDef(name="image", rank=4, dtype="float32")
  assert getattr(p, "rank") == 4
  assert getattr(p, "dtype") == "float32"


@pytest.fixture
def fuzzer() -> InputFuzzer:
  """Execute fuzzer helper logic.

  Returns:
      Computed test result or fixture value.
  """
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzzer_respects_dtype_int64(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of fuzzer respects dtype int64.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  constraints: Dict[str, Dict[str, Any]] = {"x": {"dtype": "int64"}}
  strats: Dict[str, st.SearchStrategy[Any]] = fuzzer.build_strategies(["x"], constraints=constraints)
  inputs: Dict[str, Any] = data.draw(st.fixed_dictionaries(strats))
  arr: np.ndarray[typing.Any, typing.Any] = inputs["x"]
  assert isinstance(arr, np.ndarray)
  assert getattr(arr, "dtype") == np.int64


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzzer_respects_dtype_float16(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of fuzzer respects dtype float16.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  constraints: Dict[str, Dict[str, Any]] = {"x": {"dtype": "float16"}}
  strats: Dict[str, st.SearchStrategy[Any]] = fuzzer.build_strategies(["x"], constraints=constraints)
  inputs: Dict[str, Any] = data.draw(st.fixed_dictionaries(strats))
  arr: np.ndarray[typing.Any, typing.Any] = inputs["x"]
  assert getattr(arr, "dtype") == np.float16


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzzer_respects_dtype_bool(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of fuzzer respects dtype boolean.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  constraints: Dict[str, Dict[str, Any]] = {"mask": {"dtype": "bool"}}
  strats: Dict[str, st.SearchStrategy[Any]] = fuzzer.build_strategies(["mask"], constraints=constraints)
  inputs: Dict[str, Any] = data.draw(st.fixed_dictionaries(strats))
  arr: np.ndarray[typing.Any, typing.Any] = inputs["mask"]
  assert getattr(arr, "dtype") == np.bool_


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzzer_dtype_priority_over_heuristic(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of fuzzer dtype priority over heuristic.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  constraints: Dict[str, Dict[str, Any]] = {"mask": {"dtype": "float32"}}
  strats: Dict[str, st.SearchStrategy[Any]] = fuzzer.build_strategies(["mask"], constraints=constraints)
  inputs: Dict[str, Any] = data.draw(st.fixed_dictionaries(strats))
  arr: np.ndarray[typing.Any, typing.Any] = inputs["mask"]
  assert getattr(arr, "dtype") == np.float32


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzzer_dtype_with_symbolic_shape(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of fuzzer dtype with symbolic shape.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: Dict[str, str] = {"x": "Array['N']"}
  constraints: Dict[str, Dict[str, Any]] = {"x": {"dtype": "int32"}}
  strats: Dict[str, st.SearchStrategy[Any]] = fuzzer.build_strategies(["x"], hints=hints, constraints=constraints)
  inputs: Dict[str, Any] = data.draw(st.fixed_dictionaries(strats))
  arr: np.ndarray[typing.Any, typing.Any] = inputs["x"]
  assert getattr(arr, "dtype") == np.int32
  assert len(getattr(arr, "shape")) == 1
