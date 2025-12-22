"""
Tests for ODL Schema Extension: Dtype Constraints.
Corresponds to Limitation #2 in the Architectural roadmap.

Verifies:
1. ParameterDef supports `dtype` field.
2. Fuzzer logic in InputFuzzer respects this constraint when generating arrays.
"""

import pytest
import numpy as np
from pydantic import ValidationError
from ml_switcheroo.core.dsl import ParameterDef, OperationDef, FrameworkVariant
from ml_switcheroo.testing.fuzzer import InputFuzzer


# --- Part 1: Schema Validation Tests ---


def test_param_dtype_field_initialization():
  """
  Verify that the 'dtype' field can be initialized explicitly.
  """
  p = ParameterDef(name="idx", dtype="int64")
  assert p.dtype == "int64"
  assert p.name == "idx"


def test_param_dtype_default_is_none():
  """
  Verify 'dtype' defaults to None.
  """
  p = ParameterDef(name="x")
  assert p.dtype is None


def test_param_dtype_valid_types():
  """
  Verify that 'dtype' accepts string values.
  """
  # Should accept numpy-style string names
  p1 = ParameterDef(name="mask", dtype="bool")
  assert p1.dtype == "bool"

  p2 = ParameterDef(name="embedding", dtype="float16")
  assert p2.dtype == "float16"


def test_param_dtype_and_rank():
  """
  Verify coexistence with Rank constraint.
  """
  p = ParameterDef(name="image", rank=4, dtype="float32")
  assert p.rank == 4
  assert p.dtype == "float32"


# --- Part 2: Fuzzer Integration Logic Tests ---


@pytest.fixture
def fuzzer():
  return InputFuzzer()


def test_fuzzer_respects_dtype_int64(fuzzer):
  """
  Scenario: Constraint declares dtype='int64'.
  Expectation: Generated array is int64, even if default is float32 or int32.
  """
  constraints = {"x": {"dtype": "int64"}}
  # Heuristic for 'x' normally yields float32
  inputs = fuzzer.generate_inputs(["x"], constraints=constraints)

  arr = inputs["x"]
  assert isinstance(arr, np.ndarray)
  assert arr.dtype == np.int64


def test_fuzzer_respects_dtype_float16(fuzzer):
  """
  Scenario: Constraint declares dtype='float16'.
  """
  constraints = {"x": {"dtype": "float16"}}
  inputs = fuzzer.generate_inputs(["x"], constraints=constraints)

  arr = inputs["x"]
  assert arr.dtype == np.float16


def test_fuzzer_respects_dtype_bool(fuzzer):
  """
  Scenario: Constraint declares dtype='bool'.
  """
  constraints = {"mask": {"dtype": "bool"}}
  inputs = fuzzer.generate_inputs(["mask"], constraints=constraints)

  arr = inputs["mask"]
  assert arr.dtype == bool


def test_fuzzer_dtype_priority_over_heuristic(fuzzer):
  """
  Scenario: Name 'mask' implies bool via heuristic.
            Constraint 'dtype' explicitly asks for 'float32' (e.g. attention mask floats).
  Expectation: Float32 wins.
  """
  constraints = {"mask": {"dtype": "float32"}}
  inputs = fuzzer.generate_inputs(["mask"], constraints=constraints)

  arr = inputs["mask"]
  assert arr.dtype == np.float32


def test_fuzzer_dtype_with_symbolic_shape(fuzzer):
  """
  Scenario: Typed hint with symbolic shape AND dtype constraint.
  """
  hints = {"x": "Array['N']"}
  constraints = {"x": {"dtype": "int32"}}
  inputs = fuzzer.generate_inputs(["x"], hints=hints, constraints=constraints)

  arr = inputs["x"]
  assert arr.dtype == np.int32
  assert len(arr.shape) == 1
