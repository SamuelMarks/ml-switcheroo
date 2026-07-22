"""Test suite for the Runner module."""

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


def test_runner_shape_calc_success(mock_frameworks):
  """Verifies the behavior of runner shape calculation successfully."""
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}}
  (pass_ok, msg) = runner.verify(variants, params=["x"], shape_calc="lambda x: ()")
  assert pass_ok


def test_runner_shape_calc_mismatch(mock_frameworks):
  """Verifies the behavior of runner shape calculation mismatch."""
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}}
  (pass_ok, msg) = runner.verify(variants, params=["x"], shape_calc="lambda x: (1, 2)")
  assert not pass_ok
  assert "Shape Mismatch" in msg


def test_runner_shape_calc_error(mock_frameworks):
  """Verifies the behavior of runner shape calculation correctly handling an error."""
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}}
  (pass_ok, msg) = runner.verify(variants, params=["x"], shape_calc="lambda x:")
  assert not pass_ok
  assert "Shape Calculation Error" in msg


def test_runner_crash_recovery(mock_frameworks):
  """Verifies the behavior of runner crash recovery."""
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.bad_api"}}
  mock_frameworks["torch"].bad_api.side_effect = Exception("Mock Crash")
  (pass_ok, msg) = runner.verify(variants, params=["x"])
  assert not pass_ok
  assert "Crash in torch" in msg
