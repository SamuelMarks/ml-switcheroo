"""Test suite for the Runtime Comparator Exact module."""

import numpy as np
from ml_switcheroo.generated_tests.runtime import verify_results


def test_exact_mode_passes_identical():
  """Verifies the behavior of exact mode passes identical."""
  a = np.array([1.0, 2.0])
  assert verify_results(a, a, exact=True)


def test_exact_mode_fails_approx():
  """Verifies the behavior of exact mode fails approx."""
  a = np.array([1.0])
  b = np.array([1.000000001])
  assert verify_results(a, b, rtol=1e-05, exact=False)
  assert not verify_results(a, b, exact=True)


def test_exact_mode_bools():
  """Verifies the behavior of exact mode bools."""
  a = np.array([True, False])
  b = np.array([True, True])
  assert not verify_results(a, b, exact=True)
  assert verify_results(a, a, exact=True)


def test_exact_mode_recursion():
  """Verifies the behavior of exact mode recursion."""
  a = [np.array([1.0]), {"k": np.array([2.0])}]
  b = [np.array([1.000001]), {"k": np.array([2.0])}]
  assert not verify_results(a, b, exact=True)
  assert verify_results(a, a, exact=True)
