"""
Tests for List[Tensor] Layout Consistency.

Verifies that `List[Array]` generation produces lists where all elements
share the same shape, enabling successful stacking/concatenation tests.
"""

import pytest
import numpy as np
import random
from ml_switcheroo.testing.fuzzer.parser import generate_from_hint


@pytest.fixture
def empty_context():
  return {}, {}


def test_list_array_consistency(empty_context):
  """
  Scenario: Hint is 'List[Array]'.
  Expectation: All arrays in the list have identical shapes.
  """
  symbol_map, constraints = empty_context
  base_shape = (2, 3)

  # We call it multiple times to ensure random logic is stable
  for _ in range(10):
    val = generate_from_hint(
      "List[Array]", base_shape=base_shape, depth=0, max_depth=3, symbol_map=symbol_map, constraints=constraints
    )

    assert isinstance(val, list)
    assert len(val) >= 2

    first_shape = val[0].shape
    for item in val[1:]:
      assert item.shape == first_shape, f"Shape mismatch: {item.shape} != {first_shape}"


def test_list_symbolic_consistency(empty_context):
  """
  Scenario: Hint is "List[Array['N']]".
  Expectation: Should be consistent via symbol_map already, but verified here.
  """
  symbol_map, constraints = empty_context
  val = generate_from_hint(
    "List[Array['N']]", base_shape=(1,), depth=0, max_depth=3, symbol_map=symbol_map, constraints=constraints
  )

  first_shape = val[0].shape
  for item in val[1:]:
    assert item.shape == first_shape


def test_list_primitives_ignored(empty_context):
  """
  Scenario: 'List[int]'.
  Expectation: Consistency logic shouldn't crash or enforce meaningless constraints on ints.
  """
  val = generate_from_hint("List[int]", base_shape=(1,), depth=0, max_depth=3, symbol_map={}, constraints={})
  assert isinstance(val, list)
  assert isinstance(val[0], int)
