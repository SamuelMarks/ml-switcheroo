"""Test suite for the Runner Exec module."""

import sys
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from ml_switcheroo.testing.runner import EquivalenceRunner
from ml_switcheroo.frameworks.numpy import NumpyAdapter


@pytest.fixture
def mock_frameworks():
  """Provides a mock frameworks for testing."""

  def create_safe_mock(name, ret_val=5.0):
    """Creates safe mock."""
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


def test_runner_uses_adapter_registry_for_normalization(mock_frameworks):
  """Verifies the behavior of runner uses adapter registry for normalization."""
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}}
  mock_adapter = MagicMock()
  mock_adapter.convert.return_value = "normalized_via_adapter"
  with patch("ml_switcheroo.testing.runner.get_adapter") as mock_get:
    mock_get.return_value = mock_adapter
    runner.verify(variants, params=["x"])
    mock_get.assert_called_with("numpy")
    mock_adapter.convert.assert_called()


def test_equivalence_flow_integration(mock_frameworks):
  """Verifies the behavior of equivalence flow integration."""
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}, "jax": {"api": "jax.numpy.sum"}}
  (pass_ok, msg) = runner.verify(variants, params=["x"])
  assert pass_ok
  assert "✅ Verified" in msg


def test_adapter_normalization_logic_real():
  """Verifies the behavior of adapter normalization logic real."""
  adapter = NumpyAdapter()
  mock_tensor = MagicMock()
  mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(1.0)
  assert adapter.convert(mock_tensor) == np.array(1.0)
  arr = np.array([1, 2])
  assert np.allclose(adapter.convert(arr), arr)
  data = [mock_tensor, arr]
  converted = adapter.convert(data)
  assert isinstance(converted, list)
  assert converted[0] == np.array(1.0)
  assert np.allclose(converted[1], arr)
  data_dict = {"k": mock_tensor}
  converted_dict = adapter.convert(data_dict)
  assert converted_dict["k"] == np.array(1.0)


def test_deep_compare_logic_robustness():
  """Verifies the behavior of deep compare logic robustness."""
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


def test_deep_compare_nan_handling():
  """Verifies the behavior of deep compare nan handling."""
  runner = EquivalenceRunner()
  a = np.array([1.0, np.nan])
  b = np.array([1.0, np.nan])
  assert runner._deep_compare(a, b)
  c = np.array([1.0, 0.0])
  assert not runner._deep_compare(a, c)


def test_argument_renaming_application(mock_frameworks):
  """Verifies the behavior of argument renaming application."""
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum", "args": {"axis": "dim"}}}
  with patch.object(runner, "_execute_api") as mock_exec:
    runner.verify(variants, params=["axis"], hints={"axis": "int"})
    (args, kwargs) = mock_exec.call_args
    passed_kwargs = args[1]
    assert "dim" in passed_kwargs
    assert "axis" not in passed_kwargs


def test_crash_reporting(mock_frameworks):
  """Verifies the behavior of crash reporting."""
  runner = EquivalenceRunner()
  mock_frameworks["torch"].sum.side_effect = ValueError("Mock Crash")
  variants = {"torch": {"api": "torch.sum"}}
  (passed, msg) = runner.verify(variants, params=["x"])
  assert not passed
  assert "Crash in torch" in msg
  assert "Mock Crash" in msg
