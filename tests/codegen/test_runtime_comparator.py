"""Test suite for the Runtime Comparator module."""

import numpy as np
from ml_switcheroo.generated_tests.runtime import verify_results


def test_compare_simple_arrays():
  """Verifies the behavior of compare simple arrays."""
  a = np.array([1.0, 2.0])
  b = np.array([1.000001, 2.0])
  assert verify_results(a, b)
  c = np.array([5.0, 2.0])
  assert not verify_results(a, c)


def test_compare_shapes_mismatch():
  """Verifies the behavior of compare shapes mismatch."""
  a = np.ones((2, 2))
  b = np.ones((2, 3))
  assert not verify_results(a, b)


def test_compare_list_of_tensors():
  """Verifies the behavior of compare list of tensors."""
  a = [np.zeros(2), np.ones(2)]
  b = [np.zeros(2), np.ones(2)]
  assert verify_results(a, b)
  c = [np.zeros(2), np.zeros(2)]
  assert not verify_results(a, c)


def test_compare_tuple_structure():
  """Verifies the behavior of compare tuple structure."""
  a = (np.array(1), {"key": np.array(2)})
  b = (np.array(1), {"key": np.array(2)})
  assert verify_results(a, b)
  c = [np.array(1), {"key": np.array(2)}]
  assert verify_results(a, c)
  d = (np.array(1),)
  assert not verify_results(a, d)


def test_compare_dict_mismatch():
  """Verifies the behavior of compare dictionary mismatch."""
  a = {"x": 1}
  b = {"y": 1}
  assert not verify_results(a, b)
  c = {"x": 2}
  assert not verify_results(a, c)


def test_compare_boolean_exact():
  """Verifies the behavior of compare boolean exact."""
  a = np.array([True, False])
  b = np.array([True, True])
  assert not verify_results(a, b)
  a2 = np.array([True, False])
  assert verify_results(a, a2)


def test_compare_nan_handling():
  """Verifies the behavior of compare nan handling."""
  a = np.array([np.nan, 1.0])
  b = np.array([np.nan, 1.0])
  assert verify_results(a, b)
  c = np.array([0.0, 1.0])
  assert not verify_results(a, c)


def test_compare_strings():
  """Verifies the behavior of compare strings."""
  a = "same"
  b = "same"
  assert verify_results(a, b)
  assert not verify_results(a, "diff")


def test_compare_none():
  """Verifies the behavior of compare none."""
  assert verify_results(None, None)
  assert not verify_results(None, 1)


def test_compare_scalar_vs_0d_array():
  """Verifies the behavior of compare scalar vs 0d array."""
  assert verify_results(1.0, np.array(1.0))
