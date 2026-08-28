"""Test suite for the Runtime Comparator module."""

import typing
import numpy as np
from ml_switcheroo.generated_tests.runtime import verify_results


def test_compare_simple_arrays() -> None:
  """Verifies the behavior of compare simple arrays."""
  a: np.ndarray = np.array([1.0, 2.0])
  b: np.ndarray = np.array([1.000001, 2.0])
  assert verify_results(a, b)
  c: np.ndarray = np.array([5.0, 2.0])
  assert not verify_results(a, c)


def test_compare_shapes_mismatch() -> None:
  """Verifies the behavior of compare shapes mismatch."""
  a: np.ndarray = np.ones((2, 2))
  b: np.ndarray = np.ones((2, 3))
  assert not verify_results(a, b)


def test_compare_list_of_tensors() -> None:
  """Verifies the behavior of compare list of tensors."""
  a: list[np.ndarray] = [np.zeros(2), np.ones(2)]
  b: list[np.ndarray] = [np.zeros(2), np.ones(2)]
  assert verify_results(a, b)
  c: list[np.ndarray] = [np.zeros(2), np.zeros(2)]
  assert not verify_results(a, c)


def test_compare_tuple_structure() -> None:
  """Verifies the behavior of compare tuple structure."""
  a: tuple[np.ndarray, dict[str, np.ndarray]] = (np.array(1), {"key": np.array(2)})
  b: tuple[np.ndarray, dict[str, np.ndarray]] = (np.array(1), {"key": np.array(2)})
  assert verify_results(a, b)
  c: list[typing.Union[np.ndarray, dict[str, np.ndarray]]] = [np.array(1), {"key": np.array(2)}]
  assert verify_results(a, c)
  d: tuple[np.ndarray] = (np.array(1),)
  assert not verify_results(a, d)


def test_compare_dict_mismatch() -> None:
  """Verifies the behavior of compare dictionary mismatch."""
  a: dict[str, int] = {"x": 1}
  b: dict[str, int] = {"y": 1}
  assert not verify_results(a, b)
  c: dict[str, int] = {"x": 2}
  assert not verify_results(a, c)


def test_compare_boolean_exact() -> None:
  """Verifies the behavior of compare boolean exact."""
  a: np.ndarray = np.array([True, False])
  b: np.ndarray = np.array([True, True])
  assert not verify_results(a, b)
  a2: np.ndarray = np.array([True, False])
  assert verify_results(a, a2)


def test_compare_nan_handling() -> None:
  """Verifies the behavior of compare nan handling."""
  a: np.ndarray = np.array([np.nan, 1.0])
  b: np.ndarray = np.array([np.nan, 1.0])
  assert verify_results(a, b)
  c: np.ndarray = np.array([0.0, 1.0])
  assert not verify_results(a, c)


def test_compare_strings() -> None:
  """Verifies the behavior of compare strings."""
  a: str = "same"
  b: str = "same"
  assert verify_results(a, b)
  assert not verify_results(a, "diff")


def test_compare_none() -> None:
  """Verifies the behavior of compare none."""
  assert verify_results(None, None)
  assert not verify_results(None, 1)


def test_compare_scalar_vs_0d_array() -> None:
  """Verifies the behavior of compare scalar vs 0d array."""
  assert verify_results(1.0, np.array(1.0))
