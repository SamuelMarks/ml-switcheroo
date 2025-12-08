"""
Tests for Symbolic Shape Constraints in InputFuzzer.

Verifies that:
1. `Array['B', 'N']` generates consistent shapes across arguments.
2. Fixed integers in hints (`Array[32, 'N']`) are respected.
3. Symbols are resolved consistently within a single `generate_inputs` call.
"""

import pytest
import numpy as np
from ml_switcheroo.testing.fuzzer import InputFuzzer


@pytest.fixture
def fuzzer():
  return InputFuzzer()


def test_symbolic_sharing(fuzzer):
  """
  Scenario: Two inputs `x` and `y` share a dimension `N`.
  Hint: x: Array['N'], y: Array['N']
  Expect: x.shape == y.shape
  """
  hints = {"x": "Array['N']", "y": "Array['N']"}
  inputs = fuzzer.generate_inputs(["x", "y"], hints=hints)

  x = inputs["x"]
  y = inputs["y"]

  assert isinstance(x, np.ndarray)
  assert isinstance(y, np.ndarray)
  assert x.shape == y.shape
  assert len(x.shape) == 1


def test_matmul_constraints(fuzzer):
  """
  Scenario: Matmul (A, B) @ (B, C) -> (A, C).
  Hint: x: Array['A', 'B'], y: Array['B', 'C']
  Expect: x.shape[1] == y.shape[0]
  """
  hints = {"x": "Array['A', 'B']", "y": "Array['B', 'C']"}
  inputs = fuzzer.generate_inputs(["x", "y"], hints=hints)

  x = inputs["x"]
  y = inputs["y"]

  assert len(x.shape) == 2
  assert len(y.shape) == 2
  # Inner dimension must match
  assert x.shape[1] == y.shape[0]

  # A and C might coincidentally match B if random range is small,
  # but structurally B must be consistent.


def test_fixed_dimension(fuzzer):
  """
  Scenario: Fixed dimension mixed with symbolic.
  Hint: x: Array[3, 'D']
  Expect: shape[0] == 3.
  """
  hints = {"x": "Array[3, 'D']"}
  inputs = fuzzer.generate_inputs(["x"], hints=hints)

  x = inputs["x"]
  assert x.shape[0] == 3
  assert len(x.shape) == 2


def test_symbolic_list_consistency(fuzzer):
  """
  Scenario: List of arrays sharing a symbol.
  Hint: x: List[Array['Z']]
  Expect: All arrays in list have same shape (Z,).
  """
  # Note: _generate_from_hint passes symbol_map down.
  # So 'Z' is resolved once for the list generation?
  # Yes, symbol_map is shared for the entire generate_inputs call context.

  hints = {"x": "List[Array['Z']]"}
  inputs = fuzzer.generate_inputs(["x"], hints=hints)

  lst = inputs["x"]
  # Fuzzer generates list length 1-3.
  if not lst:
    return  # Valid empty list

  shape0 = lst[0].shape
  for arr in lst[1:]:
    assert arr.shape == shape0


def test_independent_calls_are_independent(fuzzer):
  """
  Scenario: Two separate calls to generate_inputs regarding 'N'.
  Expect: 'N' can be different between calls.
  """
  hints = {"x": "Array['N']"}

  # Call 1
  res1 = fuzzer.generate_inputs(["x"], hints=hints)
  s1 = res1["x"].shape[0]

  # Call 2
  res2 = fuzzer.generate_inputs(["x"], hints=hints)
  s2 = res2["x"].shape[0]

  # It's possible s1 == s2 randomly, but logic shouldn't enforce it globally.
  # We just verify no crash and they are valid arrays.
  assert isinstance(res1["x"], np.ndarray)
  assert isinstance(res2["x"], np.ndarray)


def test_tensor_alias_support(fuzzer):
  """
  Verify 'Tensor' keyword works same as 'Array'.
  """
  hints = {"x": "Tensor['A']"}
  inputs = fuzzer.generate_inputs(["x"], hints=hints)
  assert isinstance(inputs["x"], np.ndarray)
  assert len(inputs["x"].shape) == 1
