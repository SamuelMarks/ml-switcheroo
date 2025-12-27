"""
Tests for the EquivalenceRunner.

Verifies:
1.  Execution flow across mocked frameworks.
2.  **Argument Renaming**: Ensures params defined in JSON maps are applied.
3.  **Registry Integration**: Ensures `get_adapter("numpy")` is used for output normalization.
4.  Comparison Logic: Checks robustness against mismatched results and crashes.
"""

import sys
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from ml_switcheroo.testing.runner import EquivalenceRunner
from ml_switcheroo.frameworks.numpy import NumpyAdapter


@pytest.fixture
def mock_frameworks():
  """
  Injects mocks directly into sys.modules.
  """

  def create_safe_mock(name, ret_val=5.0):
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
  """
  Feature Verification: Ensure `_to_numpy` logic is gone and replaced by
  a call to the NumpyAdapter from the registry.
  """
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}}

  # Create a mock adapter to verifying it gets called
  mock_adapter = MagicMock()
  mock_adapter.convert.return_value = "normalized_via_adapter"

  with patch("ml_switcheroo.testing.runner.get_adapter") as mock_get:
    mock_get.return_value = mock_adapter

    runner.verify(variants, params=["x"])

    # Verify we requested the numpy adapter
    mock_get.assert_called_with("numpy")
    # Verify the adapter converted the result
    mock_adapter.convert.assert_called()


def test_equivalence_flow_integration(mock_frameworks):
  """
  Verify full end-to-end flow (Mock -> Adapter -> Compare).
  """
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}, "jax": {"api": "jax.numpy.sum"}}

  # Mock return values (arrays) which need normalization
  # Inherently NumpyAdapter handles arrays, so this checks default behavior
  pass_ok, msg = runner.verify(variants, params=["x"])

  assert pass_ok
  assert "Output Matched" in msg


def test_adapter_normalization_logic_real():
  """
  Verify NumpyAdapter logic handles the types previously handled by _to_numpy.
  """
  adapter = NumpyAdapter()

  # 1. Detach pattern (Torch-like)
  mock_tensor = MagicMock()
  mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(1.0)

  assert adapter.convert(mock_tensor) == np.array(1.0)

  # 2. Numpy/JAX pattern
  arr = np.array([1, 2])
  assert np.allclose(adapter.convert(arr), arr)

  # 3. Recursive List
  data = [mock_tensor, arr]
  converted = adapter.convert(data)
  assert isinstance(converted, list)
  assert converted[0] == np.array(1.0)
  assert np.allclose(converted[1], arr)

  # 4. Dict
  data_dict = {"k": mock_tensor}
  converted_dict = adapter.convert(data_dict)
  assert converted_dict["k"] == np.array(1.0)


def test_deep_compare_logic_robustness():
  """Verify comparison logic handles strings and mixed types."""
  runner = EquivalenceRunner()

  # Strings in arrays (numpy.array_equal path)
  s1 = np.array(["a", "b"])
  s2 = np.array(["a", "b"])
  assert runner._deep_compare(s1, s2)

  s3 = np.array(["a", "c"])
  assert not runner._deep_compare(s1, s3)

  # Primitives
  assert runner._deep_compare(1, 1)
  assert not runner._deep_compare(1, 2)
