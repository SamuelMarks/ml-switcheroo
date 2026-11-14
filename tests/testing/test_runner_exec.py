"""Test suite for the Runner Exec module."""

import sys
import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo.testing.runner import EquivalenceRunner


@pytest.fixture
def mock_frameworks() -> typing.Generator[dict[str, MagicMock], None, None]:
  """Create a configured `mock_frameworks` fixture for runner exec tests.

  Yields:
      Configured `mock_frameworks` fixture value (typing.Generator[dict[str, MagicMock], None, None]).
  """

  def create_safe_mock(name: str, ret_val: float = 5.0) -> MagicMock:
    """Creates safe mock.

    Args:
        name: Identifier or operation name string.
        ret_val: Input `ret_val` parameter value (`float`).

    Returns:
        Resulting `MagicMock` from create safe mock.
    """
    m = MagicMock(name=name)
    m.__iter__.side_effect = TypeError(f"'{name}' object is not iterable")
    m.return_value = ret_val
    return m

  mock_torch = create_safe_mock("torch")
  mock_torch.sum.return_value = np.array(5.0)
  mock_jax = create_safe_mock("jax")
  mock_jax_numpy = create_safe_mock("jax.numpy")
  mock_jax.numpy = mock_jax_numpy
  mock_jax.numpy.sum.return_value = np.array(5.0)
  overrides = {"torch": mock_torch, "jax": mock_jax, "jax.numpy": mock_jax_numpy}
  with patch.dict(sys.modules, overrides):
    yield overrides


def test_runner_uses_adapter_registry_for_normalization(mock_frameworks: MagicMock) -> None:
  """Verify `EquivalenceRunner` handles runner uses adapter registry for normalization as expected.

  Args:
      mock_frameworks: Input `mock_frameworks` parameter value (`MagicMock`).
  """
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}}
  mock_adapter = MagicMock()
  mock_adapter.convert.return_value = "normalized_via_adapter"
  with patch("ml_switcheroo.testing.runner.get_adapter") as mock_get:
    mock_get.return_value = mock_adapter
    runner.verify(variants, params=["x"])
    mock_get.assert_called_with("numpy")
    mock_adapter.convert.assert_called()


def test_equivalence_flow_integration(mock_frameworks: MagicMock) -> None:
  """Verify `EquivalenceRunner` emits `✅ Verified` for equivalence flow integration.

  Args:
      mock_frameworks: Input `mock_frameworks` parameter value (`MagicMock`).
  """
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}, "jax": {"api": "jax.numpy.sum"}}
  pass_ok, msg = runner.verify(variants, params=["x"])
  assert pass_ok
  assert "✅ Verified" in msg


def test_adapter_normalization_logic_real() -> None:
  """Verify `NumpyAdapter` handles adapter normalization logic real as expected."""

  class _DetachableTensorStub:
    """Typed tensor stub implementing ``detach().cpu().numpy()``."""

    def detach(self) -> "_DetachableTensorStub":
      """Return detached tensor stub.

      Returns:
          Self instance.
      """
      return self

    def cpu(self) -> "_DetachableTensorStub":
      """Return CPU tensor stub.

      Returns:
          Self instance.
      """
      return self

    def numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
      """Convert stub tensor to NumPy array.

      Returns:
          Scalar float64 array ``1.0``.
      """
      return np.array(1.0)

  adapter = NumpyAdapter()
  mock_tensor = _DetachableTensorStub()
  assert adapter.convert(mock_tensor) == np.array(1.0)
  arr = np.array([1, 2])
  conv_arr = adapter.convert(arr)
  assert isinstance(conv_arr, np.ndarray)
  assert np.allclose(conv_arr, arr)
  data = [mock_tensor, arr]
  converted = adapter.convert(data)
  assert isinstance(converted, list)
  assert converted[0] == np.array(1.0)
  conv_elem = converted[1]
  assert isinstance(conv_elem, np.ndarray)
  assert np.allclose(conv_elem, arr)
  data_dict = {"k": mock_tensor}
  converted_dict = adapter.convert(data_dict)
  assert isinstance(converted_dict, dict)
  assert converted_dict["k"] == np.array(1.0)


def test_deep_compare_logic_robustness() -> None:
  """Verify `EquivalenceRunner` handles deep compare logic robustness as expected."""
  runner = EquivalenceRunner()
  s1 = np.array(["a", "b"])
  s2 = np.array(["a", "b"])
  assert runner._deep_compare(s1, s2)
  s3 = np.array(["a", "c"])
  assert not runner._deep_compare(s1, s3)
  assert runner._deep_compare(1, 1)
  assert not runner._deep_compare(1, 2)
  a = np.ones((2, 2))
  b = np.ones((2, 3))
  assert not runner._deep_compare(a, b)


def test_deep_compare_nan_handling() -> None:
  """Verify `EquivalenceRunner` handles deep compare nan handling as expected."""
  runner = EquivalenceRunner()
  a = np.array([1.0, np.nan])
  b = np.array([1.0, np.nan])
  assert runner._deep_compare(a, b)
  c = np.array([1.0, 0.0])
  assert not runner._deep_compare(a, c)


def test_argument_renaming_application(mock_frameworks: MagicMock) -> None:
  """Verify `EquivalenceRunner` emits `dim` for argument renaming application.

  Args:
      mock_frameworks: Input `mock_frameworks` parameter value (`MagicMock`).
  """
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum", "args": {"axis": "dim"}}}
  with patch.object(runner, "_execute_api") as mock_exec:
    runner.verify(variants, params=["axis"], hints={"axis": "int"})
    args, _kwargs = mock_exec.call_args
    passed_kwargs = args[1]
    assert "dim" in passed_kwargs
    assert "axis" not in passed_kwargs


def test_crash_reporting(mock_frameworks: MagicMock) -> None:
  """Verify `EquivalenceRunner` emits `Crash in torch` for crash reporting.

  Args:
      mock_frameworks: Input `mock_frameworks` parameter value (`MagicMock`).
  """
  runner = EquivalenceRunner()
  mock_frameworks["torch"].sum.side_effect = ValueError("Mock Crash")
  variants = {"torch": {"api": "torch.sum"}}
  passed, msg = runner.verify(variants, params=["x"])
  assert not passed
  assert "Crash in torch" in msg
  assert "Mock Crash" in msg


def test_runner_skip_invalid_variants(mock_frameworks: MagicMock) -> None:
  """Verifies that invalid variants are skipped gracefully.

  Args:
      mock_frameworks: Input `mock_frameworks` parameter value (`MagicMock`).
  """
  runner = EquivalenceRunner()
  # Include invalid details that aren't dicts or lack 'api'
  variants = {"torch": {"api": "torch.sum"}, "jax": "not_a_dict", "flax": {"other": "stuff"}}
  # This shouldn't crash
  passed, msg = runner.verify(variants, params=["x"])
  assert passed
  assert "✅ Verified" in msg
