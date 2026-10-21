"""Test suite for the Runner module."""

import sys
import typing
from unittest.mock import MagicMock, patch

import numpy as np
from pydantic import JsonValue
import pytest

from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo.testing.runner import EquivalenceRunner


@pytest.fixture
def mock_frameworks() -> typing.Generator[dict[str, MagicMock], None, None]:
  """Execute mock frameworks helper logic.

  Returns:
      Computed test result or fixture value.
  """

  def create_safe_mock(name: str, ret_val: float = 5.0) -> MagicMock:
    """Creates safe mock.

    Args:
        name: Test argument or fixture.
        ret_val: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
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
  """Verifies the behavior of runner uses adapter registry for normalization.

  Args:
      mock_frameworks: Test argument or fixture.
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
  """Verifies the behavior of equivalence flow integration.

  Args:
      mock_frameworks: Test argument or fixture.
  """
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}, "jax": {"api": "jax.numpy.sum"}}
  pass_ok, msg = runner.verify(variants, params=["x"])
  assert pass_ok
  assert "✅ Verified" in msg


def test_adapter_normalization_logic_real() -> None:
  """Verifies the behavior of adapter normalization logic real."""
  adapter = NumpyAdapter()
  mock_tensor = MagicMock()
  mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(1.0)
  assert adapter.convert(mock_tensor) == np.array(1.0)
  arr = np.array([1, 2])
  assert getattr({"v": getattr({"v": np}, "__getitem__")("v").allclose}, "__getitem__")("v")(
    getattr(
      {
        "v": getattr({"v": getattr({"v": adapter}, "__getitem__")("v").convert}, "__getitem__")("v")(
          getattr({"v": arr}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v"),
    getattr({"v": arr}, "__getitem__")("v"),
  )
  data = [mock_tensor, arr]
  converted = adapter.convert(data)
  assert isinstance(converted, list)
  assert converted[0] == np.array(1.0)
  assert np.allclose(converted[1], arr)
  data_dict = {"k": mock_tensor}
  converted_dict = adapter.convert(data_dict)
  assert getattr({"v": getattr({"v": converted_dict}, "__getitem__")("v")["k"]}, "__getitem__")("v") == getattr(
    {
      "v": getattr({"v": getattr({"v": np}, "__getitem__")("v").array}, "__getitem__")("v")(
        getattr({"v": 1.0}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")


def test_deep_compare_logic_robustness() -> None:
  """Verifies the behavior of deep compare logic robustness."""
  runner = EquivalenceRunner()
  s1 = np.array(["a", "b"])
  s2 = np.array(["a", "b"])
  assert runner._deep_compare(s1, s2)
  s3 = np.array(["a", "c"])
  assert not runner._deep_compare(s1, s3)
  assert runner._deep_compare(1, 1)
  assert not runner._deep_compare(1, 2)


def test_runner_shape_calc_success(mock_frameworks: MagicMock) -> None:
  """Verifies the behavior of runner shape calculation successfully.

  Args:
      mock_frameworks: Test argument or fixture.
  """
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}}
  pass_ok, msg = runner.verify(variants, params=["x"], shape_calc="lambda x: ()")
  assert pass_ok


def test_runner_shape_calc_mismatch(mock_frameworks: MagicMock) -> None:
  """Verifies the behavior of runner shape calculation mismatch.

  Args:
      mock_frameworks: Test argument or fixture.
  """
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}}
  pass_ok, msg = runner.verify(variants, params=["x"], shape_calc="lambda x: (1, 2)")
  assert not pass_ok
  assert "Shape Mismatch" in msg


def test_runner_shape_calc_error(mock_frameworks: MagicMock) -> None:
  """Verifies the behavior of runner shape calculation correctly handling an error.

  Args:
      mock_frameworks: Test argument or fixture.
  """
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.sum"}}
  pass_ok, msg = runner.verify(variants, params=["x"], shape_calc="lambda x:")
  assert not pass_ok
  assert "Shape Calculation Error" in msg


def test_runner_crash_recovery(mock_frameworks: MagicMock) -> None:
  """Verifies the behavior of runner crash recovery.

  Args:
      mock_frameworks: Test argument or fixture.
  """
  runner = EquivalenceRunner()
  variants = {"torch": {"api": "torch.bad_api"}}
  mock_frameworks["torch"].bad_api.side_effect = RuntimeError("Mock Crash")
  pass_ok, msg = runner.verify(variants, params=["x"])
  assert not pass_ok
  assert "Crash in torch" in msg


# --- Merged from test_runner_extra.py ---


def test_runner_error_branches() -> None:
  """Verify runner error branches behavior."""
  runner = EquivalenceRunner()
  assert runner._deep_compare(1, 2) is False
  assert runner._deep_compare([1, 2], [1, 2]) is True
  assert runner._deep_compare([1], [1, 2]) is False
  assert runner._deep_compare(np.array([1.0]), np.array([1.00001])) is True
  assert runner._deep_compare(np.array([1.0]), np.array([2.0])) is False

  class BadIter:
    """Test helper class BadIter."""

    def __len__(self) -> int:
      """Execute len helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return 1

    def __iter__(self) -> typing.Iterator[int]:
      """Execute iter helper logic.

      Returns:
          Computed test result or fixture value.
      """
      raise Exception("bad")

  assert runner._deep_compare(BadIter(), [1]) is False

  res_dict = {"fw1": 1, "fw2": 1}
  runner._compare_results(res_dict, 1e-5, 1e-5, [])
  with pytest.raises(AssertionError):
    runner._compare_results({"fw1": 1, "fw2": 2}, 1e-5, 1e-5, [])

  with patch.object(runner, "_execute_api", side_effect=RuntimeError("Mock Crash")):
    res, msg = runner.verify({"tf": {"api": "tf.add"}}, ["x"])
    assert res is False
    assert "Crash" in msg


def test_runner_hypothesis_exception() -> None:
  """Verify runner hypothesis exception behavior."""
  runner = EquivalenceRunner()
  # Mock fuzzer to just return a dummy strategy
  setattr(runner.fuzzer, "build_strategies", MagicMock(return_value={"x": MagicMock()}))

  # We want execute_api to return different things for different fws
  def mock_exec(api: str, args: dict[str, JsonValue]) -> int:
    """Execute mock exec helper logic.

    Args:
        api: Test argument or fixture.
        args: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if "tf" in api:
      return 1
    return 2

  setattr(runner, "_execute_api", mock_exec)

  # Run verify with multiple fws
  res, msg = runner.verify({"tf": {"api": "tf.add"}, "torch": {"api": "torch.add"}}, ["x"])
  assert res is False
  assert "Verification Failed" in msg


def test_runner_misc_misses() -> None:
  """Verify runner misc misses behavior."""
  runner = EquivalenceRunner()

  # 93 continue
  runner.verify({"bad_fw": "not_a_dict", "bad_fw2": {}}, [])

  # 158 return None
  assert runner._execute_api("no_dots", {}) is None

  # 228 exception in .numpy()
  class BadNumpy:
    """Test helper class BadNumpy."""

    def numpy(self) -> None:
      """Execute numpy helper logic.

      Returns:
          Computed test result or fixture value.
      """
      raise Exception("bad numpy")

  assert runner._deep_compare(BadNumpy(), 1) is False

  # Line 229: array shape mismatch
  assert runner._deep_compare(np.array([1]), np.array([1, 2])) is False

  # Lines 234-235: exception during array conversion in _deep_compare
  class BadArrayObj:
    """Class raising error during __array__ conversion."""

    def __array__(self, *args: JsonValue, **kwargs: JsonValue) -> None:
      """Raise conversion error.

      Args:
          args: Test argument or fixture.
          kwargs: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      raise ValueError("cannot convert to array")

  assert runner._deep_compare(np.array([1]), BadArrayObj()) is False

  # ImportError skipped when optional framework is not installed; RuntimeError recorded as failure
  with patch.object(runner, "_execute_api", side_effect=ImportError("Missing optional fw")):
    pass_imp, _ = runner.verify({"torch": {"api": "torch.sum"}, "jax": {"api": "jax.sum"}}, ["x"])
    assert pass_imp is True

  with patch.object(runner, "_execute_api", side_effect=RuntimeError("Generic failure")):
    pass_gen, msg_gen = runner.verify({"torch": {"api": "torch.sum"}, "jax": {"api": "jax.sum"}}, ["x"])
    assert pass_gen is False
    assert "Crash in" in msg_gen

  # 127->126: result without shape attribute in shape check
  with patch.object(runner, "_execute_api", return_value=42):
    pass_shape, _ = runner.verify(
      {"torch": {"api": "torch.sum"}, "jax": {"api": "jax.sum"}}, ["x"], shape_calc="lambda x: (1,)"
    )

  # 120->136: 'x' not in inputs during shape_calc
  with patch.object(runner, "_execute_api", return_value=42):
    runner.verify({"torch": {"api": "torch.sum"}}, ["y"], shape_calc="lambda y: (1,)")
