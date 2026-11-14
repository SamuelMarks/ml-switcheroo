"""Test suite for the Dsl Dtype module."""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from ml_switcheroo.core.dsl import ParameterDef
from ml_switcheroo.testing.fuzzer import InputFuzzer


def test_param_dtype_field_initialization() -> None:
  """Verify `ParameterDef` handles param dtype field initialization as expected."""
  p: ParameterDef = ParameterDef(name="idx", dtype="int64")
  assert p.dtype == "int64"
  assert p.name == "idx"


def test_param_dtype_default_is_none() -> None:
  """Verify `ParameterDef` handles param dtype default is none as expected."""
  p: ParameterDef = ParameterDef(name="x")
  assert p.dtype is None


def test_param_dtype_valid_types() -> None:
  """Verify `ParameterDef` handles param dtype valid types as expected."""
  p1: ParameterDef = ParameterDef(name="mask", dtype="bool")
  assert p1.dtype == "bool"
  p2: ParameterDef = ParameterDef(name="embedding", dtype="float16")
  assert p2.dtype == "float16"


def test_param_dtype_and_rank() -> None:
  """Verify `ParameterDef` returns `4` for param dtype and rank."""
  p: ParameterDef = ParameterDef(name="image", rank=4, dtype="float32")
  assert p.rank == 4
  assert p.dtype == "float32"


@pytest.fixture
def fuzzer() -> InputFuzzer:
  """Create a configured `InputFuzzer` fixture for DSL dtype tests.

  Returns:
      Configured `fuzzer` fixture value (InputFuzzer).
  """
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
def test_fuzzer_respects_dtype_int64(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles fuzzer respects dtype int64 as expected.

  Args:
      fuzzer: InputFuzzer instance used to synthesize test inputs.
      data: Hypothesis DataObject or input tensor payload.
  """
  constraints = {"x": {"dtype": "int64"}}
  strats = fuzzer.build_strategies(["x"], constraints=constraints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  arr = inputs["x"]
  assert isinstance(arr, np.ndarray)
  assert arr.dtype == np.int64


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
def test_fuzzer_respects_dtype_float16(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles fuzzer respects dtype float16 as expected.

  Args:
      fuzzer: InputFuzzer instance used to synthesize test inputs.
      data: Hypothesis DataObject or input tensor payload.
  """
  constraints = {"x": {"dtype": "float16"}}
  strats = fuzzer.build_strategies(["x"], constraints=constraints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  arr = inputs["x"]
  assert isinstance(arr, np.ndarray)
  assert arr.dtype == np.float16


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
def test_fuzzer_respects_dtype_bool(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles fuzzer respects dtype bool as expected.

  Args:
      fuzzer: InputFuzzer instance used to synthesize test inputs.
      data: Hypothesis DataObject or input tensor payload.
  """
  constraints = {"mask": {"dtype": "bool"}}
  strats = fuzzer.build_strategies(["mask"], constraints=constraints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  arr = inputs["mask"]
  assert isinstance(arr, np.ndarray)
  assert arr.dtype == np.bool_


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
def test_fuzzer_dtype_priority_over_heuristic(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles fuzzer dtype priority over heuristic as expected.

  Args:
      fuzzer: InputFuzzer instance used to synthesize test inputs.
      data: Hypothesis DataObject or input tensor payload.
  """
  constraints = {"mask": {"dtype": "float32"}}
  strats = fuzzer.build_strategies(["mask"], constraints=constraints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  arr = inputs["mask"]
  assert isinstance(arr, np.ndarray)
  assert arr.dtype == np.float32


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
def test_fuzzer_dtype_with_symbolic_shape(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` returns `1` for fuzzer dtype with symbolic shape.

  Args:
      fuzzer: InputFuzzer instance used to synthesize test inputs.
      data: Hypothesis DataObject or input tensor payload.
  """
  hints: dict[str, str] = {"x": "Array['N']"}
  constraints = {"x": {"dtype": "int32"}}
  strats = fuzzer.build_strategies(["x"], hints=hints, constraints=constraints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  arr = inputs["x"]
  assert isinstance(arr, np.ndarray)
  assert arr.dtype == np.int32
  assert len(arr.shape) == 1
