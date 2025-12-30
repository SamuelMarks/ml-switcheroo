"""
Tests for verify_results (Recursive Comparator).

Ensures the runtime helper handles:
1. Lists/Tuples recursion.
2. Dictionary recursion.
3. NumPy numeric comparison.
4. Exact match for specific types.
5. Shape mismatches.
"""

import pytest
import numpy as np
from ml_switcheroo.generated_tests.runtime import verify_results


def test_compare_simple_arrays():
  a = np.array([1.0, 2.0])
  b = np.array([1.000001, 2.0])  # Close enough
  assert verify_results(a, b)

  c = np.array([5.0, 2.0])
  assert not verify_results(a, c)


def test_compare_shapes_mismatch():
  a = np.ones((2, 2))
  b = np.ones((2, 3))
  assert not verify_results(a, b)


def test_compare_list_of_tensors():
  # Scenario: split() output
  a = [np.zeros(2), np.ones(2)]
  b = [np.zeros(2), np.ones(2)]
  assert verify_results(a, b)

  c = [np.zeros(2), np.zeros(2)]
  assert not verify_results(a, c)


def test_compare_tuple_structure():
  a = (np.array(1), {"key": np.array(2)})
  b = (np.array(1), {"key": np.array(2)})
  assert verify_results(a, b)

  c = [np.array(1), {"key": np.array(2)}]  # Tuple vs List mismatch (default strictness)
  # The default verify_results treats tuple/list as interchangeable sequences via zip,
  # as strict type equality makes cross-framework testing hard (some return lists vs tuples).
  assert verify_results(a, c)

  d = (np.array(1),)  # Different length
  assert not verify_results(a, d)


def test_compare_dict_mismatch():
  a = {"x": 1}
  b = {"y": 1}
  assert not verify_results(a, b)

  c = {"x": 2}
  assert not verify_results(a, c)


def test_compare_boolean_exact():
  a = np.array([True, False])
  b = np.array([True, True])
  # Should perform exact match, not float epsilon
  assert not verify_results(a, b)

  a2 = np.array([True, False])
  assert verify_results(a, a2)


def test_compare_nan_handling():
  a = np.array([np.nan, 1.0])
  b = np.array([np.nan, 1.0])
  assert verify_results(a, b)

  c = np.array([0.0, 1.0])
  assert not verify_results(a, c)


def test_compare_strings():
  # Just in case some API returns metadata
  a = "same"
  b = "same"
  assert verify_results(a, b)
  assert not verify_results(a, "diff")


def test_compare_none():
  assert verify_results(None, None)
  assert not verify_results(None, 1)


def test_compare_scalar_vs_0d_array():
  """Verify that scalar 1.0 matches np.array(1.0)."""
  assert verify_results(1.0, np.array(1.0))
