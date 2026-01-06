"""
Tests for Hypothesis Strategy Generation.
"""

import pytest
from hypothesis import given, settings, strategies as st
import numpy as np
from ml_switcheroo.testing.fuzzer.strategies import strategies_from_spec

# Note: We must define the test functions to accept the 'data' strategy argument
# which is injected by @given if st.data() is used, or arguments matching
# the strategies passed to @given.


@given(val=strategies_from_spec("int", {"min": 1, "max": 10}))
@settings(max_examples=10)
def test_int_strategy(val):
  assert 1 <= val <= 10


@given(val=strategies_from_spec("Array", {"rank": 2, "dtype": "int"}))
@settings(max_examples=10)
def test_array_strategy(val):
  assert isinstance(val, np.ndarray)
  assert len(val.shape) == 2
  assert val.dtype == np.int32


def test_symbolic_shape_sharing():
  # Requires shared context mapping
  shared = {}

  # The strategies created here depend on shared state 'shared'
  s_x = strategies_from_spec("Array['N']", {}, shared)
  s_y = strategies_from_spec("Array['N']", {}, shared)

  # We use st.data() to draw from these dependent strategies within a single test run
  @given(data=st.data())
  @settings(max_examples=10)
  def check(data):
    x = data.draw(s_x)
    y = data.draw(s_y)
    assert x.shape == y.shape

  check()
