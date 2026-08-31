"""Test suite for the Runner module."""

import sys
from typing import Any, Dict, Generator
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


def test_runner_shape_calc_success(mock_frameworks: Dict[str, MagicMock]) -> None:
  """Verifies the behavior of runner shape calculation successfully."""
  runner: EquivalenceRunner = EquivalenceRunner()
  variants: Dict[str, Dict[str, str]] = {"torch": {"api": "torch.sum"}}
  pass_ok: bool
  msg: str
  pass_ok, msg = runner.verify(variants, params=["x"], shape_calc="lambda x: ()")
  assert pass_ok


def test_runner_shape_calc_mismatch(mock_frameworks: Dict[str, MagicMock]) -> None:
  """Verifies the behavior of runner shape calculation mismatch."""
  runner: EquivalenceRunner = EquivalenceRunner()
  variants: Dict[str, Dict[str, str]] = {"torch": {"api": "torch.sum"}}
  pass_ok: bool
  msg: str
  pass_ok, msg = runner.verify(variants, params=["x"], shape_calc="lambda x: (1, 2)")
  assert not pass_ok
  assert "Shape Mismatch" in msg


def test_runner_shape_calc_error(mock_frameworks: Dict[str, MagicMock]) -> None:
  """Verifies the behavior of runner shape calculation correctly handling an error."""
  runner: EquivalenceRunner = EquivalenceRunner()
  variants: Dict[str, Dict[str, str]] = {"torch": {"api": "torch.sum"}}
  pass_ok: bool
  msg: str
  pass_ok, msg = runner.verify(variants, params=["x"], shape_calc="lambda x:")
  assert not pass_ok
  assert "Shape Calculation Error" in msg


def test_runner_crash_recovery(mock_frameworks: Dict[str, MagicMock]) -> None:
  """Verifies the behavior of runner crash recovery."""
  runner: EquivalenceRunner = EquivalenceRunner()
  variants: Dict[str, Dict[str, str]] = {"torch": {"api": "torch.bad_api"}}
  mock_frameworks["torch"].bad_api.side_effect = Exception("Mock Crash")
  pass_ok: bool
  msg: str
  pass_ok, msg = runner.verify(variants, params=["x"])
  assert not pass_ok
  assert "Crash in torch" in msg


# --- Merged from test_runner_extra.py ---


def test_runner_error_branches() -> None:
  """Docstring."""
  runner: EquivalenceRunner = EquivalenceRunner()
  assert runner._deep_compare(1, 2) is False
  assert runner._deep_compare([1, 2], [1, 2]) is True
  assert runner._deep_compare([1], [1, 2]) is False
  assert runner._deep_compare(np.array([1.0]), np.array([1.00001])) is True
  assert runner._deep_compare(np.array([1.0]), np.array([2.0])) is False

  class BadIter:
    """Docstring."""

    def __len__(self) -> int:
      """Docstring."""
      return 1

    def __iter__(self) -> Any:
      """Docstring."""
      raise Exception("bad")

  assert runner._deep_compare(BadIter(), [1]) is False

  res_dict: Dict[str, Any] = {"fw1": 1, "fw2": 1}
  runner._compare_results(res_dict, 1e-5, 1e-5, [])
  with pytest.raises(AssertionError):
    runner._compare_results({"fw1": 1, "fw2": 2}, 1e-5, 1e-5, [])

  with patch.object(runner, "_execute_api", side_effect=Exception("Mock Crash")):
    res: bool
    msg: str
    res, msg = runner.verify({"tf": {"api": "tf.add"}}, ["x"])
    assert res is False
    assert "Crash" in msg


def test_runner_hypothesis_exception() -> None:
  """Docstring."""
  runner: EquivalenceRunner = EquivalenceRunner()
  # Mock fuzzer to just return a dummy strategy
  runner.fuzzer.build_strategies = MagicMock(return_value={"x": MagicMock()})

  # We want execute_api to return different things for different fws
  def mock_exec(api: str, args: Dict[str, Any]) -> int:
    """Docstring."""
    if "tf" in api:
      return 1
    return 2

  runner._execute_api = mock_exec

  # Run verify with multiple fws
  res: bool
  msg: str
  res, msg = runner.verify({"tf": {"api": "tf.add"}, "torch": {"api": "torch.add"}}, ["x"])
  assert res is False
  assert "Verification Failed" in msg


def test_runner_misc_misses() -> None:
  """Docstring."""
  runner: EquivalenceRunner = EquivalenceRunner()

  # 93 continue
  runner.verify({"bad_fw": "not_a_dict", "bad_fw2": {}}, [])

  # 158 return None
  assert runner._execute_api("no_dots", {}) is None

  # 228 exception in .numpy()
  class BadNumpy:
    """Docstring."""

    def numpy(self) -> Any:
      """Docstring."""
      raise Exception("bad numpy")

  assert runner._deep_compare(BadNumpy(), 1) is False
