"""
Tests for Hypothesis Strategy Generation.
"""

import pytest
from hypothesis import given, strategies as st
import numpy as np
from ml_switcheroo.testing.fuzzer.strategies import strategies_from_spec


def test_int_strategy():
  s = strategies_from_spec("int", {"min": 1, "max": 10})
  val = s.example()
  assert 1 <= val <= 10


def test_array_strategy():
  s = strategies_from_spec("Array", {"rank": 2, "dtype": "int"})
  val = s.example()
  assert isinstance(val, np.ndarray)
  assert len(val.shape) == 2
  assert val.dtype == np.int32


def test_symbolic_shape_sharing():
  # Requires shared context mapping
  shared = {}

  s_x = strategies_from_spec("Array['N']", {}, shared)
  s_y = strategies_from_spec("Array['N']", {}, shared)

  # We must draw from them in the same data context
  @given(st.data())
  def check(data):
    # Draw x then y
    x = data.draw(s_x)
    y = data.draw(s_y)
    assert x.shape == y.shape

  check()
