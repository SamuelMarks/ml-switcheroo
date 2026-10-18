"""Test suite for the Runner Exec module."""

import typing

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo.testing.runner import EquivalenceRunner


@pytest.fixture
def mock_frameworks() -> typing.Any:
  """Execute mock frameworks helper logic.

  Returns:
      Computed test result or fixture value.
  """

  def create_safe_mock(name: typing.Any, ret_val: typing.Any = 5.0) -> typing.Any:
    """Creates safe mock.

    Args:
        name: Test argument or fixture.
        ret_val: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    m: typing.Any = MagicMock(name=name)
    m.__iter__.side_effect = TypeError(f"'{name}' object is not iterable")
    m.return_value = ret_val
    return m

  mock_torch: typing.Any = create_safe_mock("torch")
  mock_torch.sum.return_value = np.array(5.0)
  mock_jax: typing.Any = create_safe_mock("jax")
  mock_jax_numpy: typing.Any = create_safe_mock("jax.numpy")
  mock_jax.numpy = mock_jax_numpy
  mock_jax.numpy.sum.return_value = np.array(5.0)
  overrides: typing.Any = {"torch": mock_torch, "jax": mock_jax, "jax.numpy": mock_jax_numpy}
  with patch.dict(sys.modules, overrides):
    yield overrides


def test_runner_uses_adapter_registry_for_normalization(mock_frameworks: typing.Any) -> None:
  """Verifies the behavior of runner uses adapter registry for normalization.

  Args:
      mock_frameworks: Test argument or fixture.
  """
  runner: typing.Any = EquivalenceRunner()
  variants: typing.Any = {"torch": {"api": "torch.sum"}}
  mock_adapter: typing.Any = MagicMock()
  mock_adapter.convert.return_value = "normalized_via_adapter"
  with patch("ml_switcheroo.testing.runner.get_adapter") as mock_get:
    mock_get.return_value = mock_adapter
    runner.verify(variants, params=["x"])
    mock_get.assert_called_with("numpy")
    mock_adapter.convert.assert_called()


def test_equivalence_flow_integration(mock_frameworks: typing.Any) -> None:
  """Verifies the behavior of equivalence flow integration.

  Args:
      mock_frameworks: Test argument or fixture.
  """
  runner: typing.Any = EquivalenceRunner()
  variants: typing.Any = {"torch": {"api": "torch.sum"}, "jax": {"api": "jax.numpy.sum"}}
  pass_ok: typing.Any
  msg: typing.Any
  pass_ok, msg = runner.verify(variants, params=["x"])
  assert pass_ok
  assert "✅ Verified" in msg


def test_adapter_normalization_logic_real() -> None:
  """Verifies the behavior of adapter normalization logic real."""
  adapter: typing.Any = NumpyAdapter()
  mock_tensor: typing.Any = MagicMock()
  mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(1.0)
  assert adapter.convert(mock_tensor) == np.array(1.0)
  arr: typing.Any = np.array([1, 2])
  assert np.allclose(adapter.convert(arr), arr)
  data: typing.Any = [mock_tensor, arr]
  converted: typing.Any = adapter.convert(data)
  assert isinstance(converted, list)
  assert converted[0] == np.array(1.0)
  assert np.allclose(converted[1], arr)
  data_dict: typing.Any = {"k": mock_tensor}
  converted_dict: typing.Any = adapter.convert(data_dict)
  assert converted_dict["k"] == np.array(1.0)


def test_deep_compare_logic_robustness() -> None:
  """Verifies the behavior of deep compare logic robustness."""
  runner: typing.Any = EquivalenceRunner()
  s1: typing.Any = np.array(["a", "b"])
  s2: typing.Any = np.array(["a", "b"])
  assert runner._deep_compare(s1, s2)
  s3: typing.Any = np.array(["a", "c"])
  assert not runner._deep_compare(s1, s3)
  assert runner._deep_compare(1, 1)
  assert not runner._deep_compare(1, 2)
  a: typing.Any = np.ones((2, 2))
  b: typing.Any = np.ones((2, 3))
  assert not runner._deep_compare(a, b)


def test_deep_compare_nan_handling() -> None:
  """Verifies the behavior of deep compare nan handling."""
  runner: typing.Any = EquivalenceRunner()
  a: typing.Any = np.array([1.0, np.nan])
  b: typing.Any = np.array([1.0, np.nan])
  assert runner._deep_compare(a, b)
  c: typing.Any = np.array([1.0, 0.0])
  assert not runner._deep_compare(a, c)


def test_argument_renaming_application(mock_frameworks: typing.Any) -> None:
  """Verifies the behavior of argument renaming application.

  Args:
      mock_frameworks: Test argument or fixture.
  """
  runner: typing.Any = EquivalenceRunner()
  variants: typing.Any = {"torch": {"api": "torch.sum", "args": {"axis": "dim"}}}
  with patch.object(runner, "_execute_api") as mock_exec:
    runner.verify(variants, params=["axis"], hints={"axis": "int"})
    args: typing.Any
    kwargs: typing.Any
    args, kwargs = mock_exec.call_args
    passed_kwargs: typing.Any = args[1]
    assert "dim" in passed_kwargs
    assert "axis" not in passed_kwargs


def test_crash_reporting(mock_frameworks: typing.Any) -> None:
  """Verifies the behavior of crash reporting.

  Args:
      mock_frameworks: Test argument or fixture.
  """
  runner: typing.Any = EquivalenceRunner()
  mock_frameworks["torch"].sum.side_effect = ValueError("Mock Crash")
  variants: typing.Any = {"torch": {"api": "torch.sum"}}
  passed: typing.Any
  msg: typing.Any
  passed, msg = runner.verify(variants, params=["x"])
  assert not passed
  assert "Crash in torch" in msg
  assert "Mock Crash" in msg


def test_runner_skip_invalid_variants(mock_frameworks: typing.Any) -> None:
  """Verifies that invalid variants are skipped gracefully.

  Args:
      mock_frameworks: Test argument or fixture.
  """
  runner: typing.Any = EquivalenceRunner()
  # Include invalid details that aren't dicts or lack 'api'
  variants: typing.Any = {"torch": {"api": "torch.sum"}, "jax": "not_a_dict", "flax": {"other": "stuff"}}
  # This shouldn't crash
  passed: typing.Any
  msg: typing.Any
  passed, msg = runner.verify(variants, params=["x"])
  assert passed
  assert "✅ Verified" in msg
