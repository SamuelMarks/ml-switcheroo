"""
Tests for the EquivalenceRunner Execution Logic.

Verifies:
1.  Execution flow across mocked frameworks.
2.  **Argument Renaming**: Ensures params defined in JSON maps are applied.
3.  **Registry Integration**: Ensures `get_adapter("numpy")` is used for output normalization.
4.  Comparison Logic: Checks robustness against mismatched results and crashes.
5.  Deep Comparison: Nested structures and shape mismatches.
"""

import sys
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from ml_switcheroo.testing.runner import EquivalenceRunner
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.frameworks.numpy import NumpyAdapter


@pytest.fixture
def mock_frameworks():
  """
  Injects mocks directly into sys.modules.
  Returns the overrides dict for further manipulation if needed.
  """

  def create_safe_mock(name, ret_val=5.0):
    # Create a mock that behaves like a module
    m = MagicMock(name=name)
    # Prevent iteration to differentiate from lists/iterables during inspection
    m.__iter__.side_effect = TypeError(f"'{name}' object is not iterable")
    return m

  # Mock 'torch.sum'
  mock_torch = create_safe_mock("torch")
  mock_torch.sum.return_value = np.array(5.0)

  # Mock 'jax.numpy.sum'
  mock_jax = create_safe_mock("jax")
  mock_jax_numpy = create_safe_mock("jax.numpy")
  mock_jax.numpy = mock_jax_numpy
  mock_jax.numpy.sum.return_value = np.array(5.0)

  overrides = {"torch": mock_torch, "jax": mock_jax, "jax.numpy": mock_jax_numpy}

  with patch.dict(sys.modules, overrides):
    yield overrides


def test_runner_uses_adapter_registry_for_normalization(mock_frameworks):
  """
  Feature Verification: Ensure `_to_numpy` logic calls the NumpyAdapter.
  """
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}}

  # Create a mock adapter to verify it gets called
  mock_adapter = MagicMock()
  mock_adapter.convert.return_value = "normalized_via_adapter"

  # Patch get_adapter. Note we patch where it is *defined* in runner's scope or imported.
  # runner.py: `from ml_switcheroo.frameworks import get_adapter`
  with patch("ml_switcheroo.testing.runner.get_adapter") as mock_get:
    mock_get.return_value = mock_adapter

    # Run verify (params=["x"] will generate random x)
    runner.verify(variants, params=["x"])

    # Verify we requested the numpy adapter
    mock_get.assert_called_with("numpy")
    # Verify the adapter converted the result
    mock_adapter.convert.assert_called()


def test_equivalence_flow_integration(mock_frameworks):
  """
  Verify full end-to-end flow (Mock -> Adapter -> Compare).
  Both frameworks return 5.0 (match).
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

  # 1. Detach pattern (Torch-like tensor)
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
  """Verify comparison logic handles strings, shapes, and mixed types."""
  runner = EquivalenceRunner()

  # Strings/Text (numpy.array_equal path)
  s1 = np.array(["a", "b"])
  s2 = np.array(["a", "b"])
  assert runner._deep_compare(s1, s2)

  s3 = np.array(["a", "c"])
  assert not runner._deep_compare(s1, s3)

  # Primitives
  assert runner._deep_compare(1, 1)
  assert not runner._deep_compare(1, 2)

  # Shape Mismatch
  a = np.ones((2, 2))
  b = np.ones((2, 3))
  assert not runner._deep_compare(a, b)


def test_deep_compare_nan_handling():
  """Verify NaN equality logic."""
  runner = EquivalenceRunner()

  a = np.array([1.0, np.nan])
  b = np.array([1.0, np.nan])

  # np.allclose(equal_nan=True) should handle this
  assert runner._deep_compare(a, b)

  c = np.array([1.0, 0.0])
  assert not runner._deep_compare(a, c)


def test_argument_renaming_application(mock_frameworks):
  """
  Verify parameter mapping logic (std -> fw).
  Scenario: Standard 'axis' -> Torch 'dim'.
  """
  runner = EquivalenceRunner()

  # Setup Variant with arg mapping
  variants = {
    "torch": {
      "api": "torch.sum",
      "args": {"axis": "dim"},  # Map std 'axis' to 'dim'
    }
  }

  # Patch execute to inspect args
  with patch.object(runner, "_execute_api") as mock_exec:
    runner.verify(variants, params=["axis"], hints={"axis": "int"})

    # Verify call args
    args, kwargs = mock_exec.call_args
    passed_kwargs = args[1]  # arg 1 is kwargs dict

    # Should contain 'dim', not 'axis'
    assert "dim" in passed_kwargs
    assert "axis" not in passed_kwargs


def test_crash_reporting(mock_frameworks):
  """Verify runner captures exceptions and returns False."""
  runner = EquivalenceRunner()

  # Force torch mock to raise error
  mock_frameworks["torch"].sum.side_effect = ValueError("Mock Crash")

  variants = {"torch": {"api": "torch.sum"}}

  passed, msg = runner.verify(variants, params=["x"])

  assert not passed
  assert "Crash in torch" in msg
  assert "ValueError: Mock Crash" in msg
