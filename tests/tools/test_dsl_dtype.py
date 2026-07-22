"""Test suite for the Dsl Dtype module."""

import pytest
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck
from ml_switcheroo.core.dsl import ParameterDef
from ml_switcheroo.testing.fuzzer import InputFuzzer


def test_param_dtype_field_initialization():
  """Verifies the behavior of parameter dtype field initialization."""
  p = ParameterDef(name="idx", dtype="int64")
  assert p.dtype == "int64"
  assert p.name == "idx"


def test_param_dtype_default_is_none():
  """Verifies the behavior of parameter dtype default is none."""
  p = ParameterDef(name="x")
  assert p.dtype is None


def test_param_dtype_valid_types():
  """Verifies the behavior of parameter dtype valid types."""
  p1 = ParameterDef(name="mask", dtype="bool")
  assert p1.dtype == "bool"
  p2 = ParameterDef(name="embedding", dtype="float16")
  assert p2.dtype == "float16"


def test_param_dtype_and_rank():
  """Verifies the behavior of parameter dtype and rank."""
  p = ParameterDef(name="image", rank=4, dtype="float32")
  assert p.rank == 4
  assert p.dtype == "float32"


@pytest.fixture
def fuzzer():
  """Provides a mock fuzzer for testing."""
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzzer_respects_dtype_int64(fuzzer, data):
  """Verifies the behavior of fuzzer respects dtype int64."""
  constraints = {"x": {"dtype": "int64"}}
  strats = fuzzer.build_strategies(["x"], constraints=constraints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  arr = inputs["x"]
  assert isinstance(arr, np.ndarray)
  assert arr.dtype == np.int64


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzzer_respects_dtype_float16(fuzzer, data):
  """Verifies the behavior of fuzzer respects dtype float16."""
  constraints = {"x": {"dtype": "float16"}}
  strats = fuzzer.build_strategies(["x"], constraints=constraints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  arr = inputs["x"]
  assert arr.dtype == np.float16


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzzer_respects_dtype_bool(fuzzer, data):
  """Verifies the behavior of fuzzer respects dtype boolean."""
  constraints = {"mask": {"dtype": "bool"}}
  strats = fuzzer.build_strategies(["mask"], constraints=constraints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  arr = inputs["mask"]
  assert arr.dtype == bool


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzzer_dtype_priority_over_heuristic(fuzzer, data):
  """Verifies the behavior of fuzzer dtype priority over heuristic."""
  constraints = {"mask": {"dtype": "float32"}}
  strats = fuzzer.build_strategies(["mask"], constraints=constraints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  arr = inputs["mask"]
  assert arr.dtype == np.float32


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzzer_dtype_with_symbolic_shape(fuzzer, data):
  """Verifies the behavior of fuzzer dtype with symbolic shape."""
  hints = {"x": "Array['N']"}
  constraints = {"x": {"dtype": "int32"}}
  strats = fuzzer.build_strategies(["x"], hints=hints, constraints=constraints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  arr = inputs["x"]
  assert arr.dtype == np.int32
  assert len(arr.shape) == 1
