"""Test suite for the Runner Exec module."""

import sys
from typing import Any, Dict, Generator, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo.testing.runner import EquivalenceRunner


@pytest.fixture
def mock_frameworks() -> Generator[Dict[str, MagicMock], None, None]:
  """Docstring."""

  def create_safe_mock(name: str, ret_val: float = 5.0) -> MagicMock:
    """Creates safe mock."""
    m: MagicMock = MagicMock(name=name)
    m.__iter__.side_effect = TypeError(f"'{name}' object is not iterable")
    m.return_value = ret_val
    return m

  mock_torch: MagicMock = create_safe_mock("torch")
  mock_torch.sum.return_value = np.array(5.0)
  mock_jax: MagicMock = create_safe_mock("jax")
  mock_jax_numpy: MagicMock = create_safe_mock("jax.numpy")
  mock_jax.numpy = mock_jax_numpy
  mock_jax.numpy.sum.return_value = np.array(5.0)
  overrides: Dict[str, MagicMock] = {"torch": mock_torch, "jax": mock_jax, "jax.numpy": mock_jax_numpy}
  with patch.dict(sys.modules, overrides):
    yield overrides


def test_runner_uses_adapter_registry_for_normalization(mock_frameworks: Dict[str, MagicMock]) -> None:
  """Verifies the behavior of runner uses adapter registry for normalization."""
  runner: EquivalenceRunner = EquivalenceRunner()
  variants: Dict[str, Dict[str, str]] = {"torch": {"api": "torch.sum"}}
  mock_adapter: MagicMock = MagicMock()
  mock_adapter.convert.return_value = "normalized_via_adapter"
  with patch("ml_switcheroo.testing.runner.get_adapter") as mock_get:
    mock_get.return_value = mock_adapter
    runner.verify(variants, params=["x"])
    mock_get.assert_called_with("numpy")
    mock_adapter.convert.assert_called()


def test_equivalence_flow_integration(mock_frameworks: Dict[str, MagicMock]) -> None:
  """Verifies the behavior of equivalence flow integration."""
  runner: EquivalenceRunner = EquivalenceRunner()
  variants: Dict[str, Dict[str, str]] = {"torch": {"api": "torch.sum"}, "jax": {"api": "jax.numpy.sum"}}
  pass_ok: bool
  msg: str
  pass_ok, msg = runner.verify(variants, params=["x"])
  assert pass_ok
  assert "✅ Verified" in msg


def test_adapter_normalization_logic_real() -> None:
  """Verifies the behavior of adapter normalization logic real."""
  adapter: NumpyAdapter = NumpyAdapter()
  mock_tensor: MagicMock = MagicMock()
  mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(1.0)
  assert adapter.convert(mock_tensor) == np.array(1.0)
  arr: np.ndarray = np.array([1, 2])
  assert np.allclose(adapter.convert(arr), arr)
  data: list = [mock_tensor, arr]
  converted: list = adapter.convert(data)
  assert isinstance(converted, list)
  assert converted[0] == np.array(1.0)
  assert np.allclose(converted[1], arr)
  data_dict: dict = {"k": mock_tensor}
  converted_dict: dict = adapter.convert(data_dict)
  assert converted_dict["k"] == np.array(1.0)


def test_deep_compare_logic_robustness() -> None:
  """Verifies the behavior of deep compare logic robustness."""
  runner: EquivalenceRunner = EquivalenceRunner()
  s1: np.ndarray = np.array(["a", "b"])
  s2: np.ndarray = np.array(["a", "b"])
  assert runner._deep_compare(s1, s2)
  s3: np.ndarray = np.array(["a", "c"])
  assert not runner._deep_compare(s1, s3)
  assert runner._deep_compare(1, 1)
  assert not runner._deep_compare(1, 2)
  a: np.ndarray = np.ones((2, 2))
  b: np.ndarray = np.ones((2, 3))
  assert not runner._deep_compare(a, b)


def test_deep_compare_nan_handling() -> None:
  """Verifies the behavior of deep compare nan handling."""
  runner: EquivalenceRunner = EquivalenceRunner()
  a: np.ndarray = np.array([1.0, np.nan])
  b: np.ndarray = np.array([1.0, np.nan])
  assert runner._deep_compare(a, b)
  c: np.ndarray = np.array([1.0, 0.0])
  assert not runner._deep_compare(a, c)


def test_argument_renaming_application(mock_frameworks: Dict[str, MagicMock]) -> None:
  """Verifies the behavior of argument renaming application."""
  runner: EquivalenceRunner = EquivalenceRunner()
  variants: Dict[str, Dict[str, Any]] = {"torch": {"api": "torch.sum", "args": {"axis": "dim"}}}
  with patch.object(runner, "_execute_api") as mock_exec:
    runner.verify(variants, params=["axis"], hints={"axis": "int"})
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    args, kwargs = mock_exec.call_args
    passed_kwargs: Dict[str, Any] = args[1]
    assert "dim" in passed_kwargs
    assert "axis" not in passed_kwargs


def test_crash_reporting(mock_frameworks: Dict[str, MagicMock]) -> None:
  """Verifies the behavior of crash reporting."""
  runner: EquivalenceRunner = EquivalenceRunner()
  mock_frameworks["torch"].sum.side_effect = ValueError("Mock Crash")
  variants: Dict[str, Dict[str, str]] = {"torch": {"api": "torch.sum"}}
  passed: bool
  msg: str
  passed, msg = runner.verify(variants, params=["x"])
  assert not passed
  assert "Crash in torch" in msg
  assert "Mock Crash" in msg


def test_runner_skip_invalid_variants(mock_frameworks: Dict[str, MagicMock]) -> None:
  """Verifies that invalid variants are skipped gracefully."""
  runner: EquivalenceRunner = EquivalenceRunner()
  # Include invalid details that aren't dicts or lack 'api'
  variants: Dict[str, Any] = {"torch": {"api": "torch.sum"}, "jax": "not_a_dict", "flax": {"other": "stuff"}}
  # This shouldn't crash
  passed: bool
  msg: str
  passed, msg = runner.verify(variants, params=["x"])
  assert passed
  assert "✅ Verified" in msg
