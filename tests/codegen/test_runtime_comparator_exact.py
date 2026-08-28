"""Test suite for the Runtime Comparator Exact module."""

import typing
import numpy as np
from ml_switcheroo.generated_tests.runtime import verify_results


def test_exact_mode_passes_identical() -> None:
  """Verifies the behavior of exact mode passes identical."""
  a: np.ndarray = np.array([1.0, 2.0])
  assert verify_results(a, a, exact=True)


def test_exact_mode_fails_approx() -> None:
  """Verifies the behavior of exact mode fails approx."""
  a: np.ndarray = np.array([1.0])
  b: np.ndarray = np.array([1.000000001])
  assert verify_results(a, b, rtol=1e-05, exact=False)
  assert not verify_results(a, b, exact=True)


def test_exact_mode_bools() -> None:
  """Verifies the behavior of exact mode bools."""
  a: np.ndarray = np.array([True, False])
  b: np.ndarray = np.array([True, True])
  assert not verify_results(a, b, exact=True)
  assert verify_results(a, a, exact=True)


def test_exact_mode_recursion() -> None:
  """Verifies the behavior of exact mode recursion."""
  a: list[typing.Union[np.ndarray, dict[str, np.ndarray]]] = [np.array([1.0]), {"k": np.array([2.0])}]
  b: list[typing.Union[np.ndarray, dict[str, np.ndarray]]] = [np.array([1.000001]), {"k": np.array([2.0])}]
  assert not verify_results(a, b, exact=True)
  assert verify_results(a, a, exact=True)
