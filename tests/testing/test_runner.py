"""Test suite for the Runner Missing module."""

import typing
from pydantic import JsonValue


def test_runner_run_exceptions() -> None:
  """Verifies the behavior of runner run exceptions."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()
  variants = {"jax": "bad"}
  res, msg = sr.verify(variants, [], {}, {})
  assert res is True
  with __import__("unittest.mock").mock.patch.object(sr, "_execute_api", return_value=1):
    res, msg = sr.verify(variants, [], {}, {})

  def force_fail(*args: JsonValue, **kwargs: JsonValue) -> None:
    """Helper to force fail.

    Args:
        args: Test argument or fixture.
        kwargs: Test argument or fixture.
    """
    raise ValueError("hypothesis failed")

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.runner.EquivalenceRunner._compare_results", side_effect=ValueError("fail hypothesis")
  ):
    res, msg = sr.verify(variants, [], {}, {})
    assert "Verification Failed" in msg


def test_runner_execute_api() -> None:
  """Verifies the behavior of runner execute API."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()
  assert sr._execute_api("no_dot", {}) is None


def test_runner_compare() -> None:
  """Verifies the behavior of runner compare."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()
  err_box: list[str] = []
  import pytest

  with pytest.raises(AssertionError):
    sr._compare_results({"jax": 1, "torch": 2}, 0.001, 0.0001, err_box)
  assert len(err_box) == 1
  with pytest.raises(AssertionError):
    sr._compare_results({"jax": [1], "torch": [1, 2]}, 0.001, 0.0001, [])


def test_runner_deep_compare_exceptions() -> None:
  """Verifies the behavior of runner deep compare exceptions."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()

  class BadNumpy:
    """Test helper class BadNumpy."""

    def __array__(self, *args: JsonValue, **kwargs: JsonValue) -> None:
      """Helper to   array  .

      Args:
          args: Test argument or fixture.
          kwargs: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      raise Exception("fail")

  assert sr._deep_compare(1, BadNumpy()) is False
  import numpy as np

  assert sr._deep_compare(np.array(["a"]), np.array(["a"])) is True
  assert sr._deep_compare(np.array(["a"]), np.array(["b"])) is False


def test_runner_run_details_not_dict() -> None:
  """Verifies the behavior of runner run details not dictionary."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()
  # Hit line 93: details is dict but missing "api" key
  variants = {"jax": "string", "torch": {"api": "torch.add"}, "mlx": {"not_api": "something"}}
  with __import__("unittest.mock").mock.patch.object(sr, "_execute_api", return_value=1):
    res, msg = sr.verify(variants, [], {}, {})
    assert "Verified" in msg or "Failures" in msg


def test_runner_deep_compare_shape_mismatch() -> None:
  """Verifies behavior when arrays have different shapes."""
  import numpy as np

  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()
  # Hit line 228
  assert sr._deep_compare(np.array([1, 2]), np.array([1, 2, 3])) is False


def test_runner_run_shape_calculation_error() -> None:
  """Verifies the behavior of runner run shape calculation correctly handling an error."""
  import hypothesis.strategies as st
  import numpy as np

  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()
  variants = {"jax": {"api": "jax.add"}}

  class DummyFuzzer:
    """Test helper class DummyFuzzer."""

    def build_strategies(
      self,
      p: list[str],
      h: typing.Optional[dict[str, str]] = None,
      c: typing.Optional[dict[str, dict[str, JsonValue]]] = None,
    ) -> dict[str, st.SearchStrategy[np.ndarray[tuple[int, ...], np.dtype[np.int64]]]]:
      """Mock implementation of build strategies.

      Args:
          p: Test argument or fixture.
          h: Test argument or fixture.
          c: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return {"x": st.just(np.array([1]))}

    def adapt_to_framework(
      self,
      args: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.int64]]],
      fw: str,
    ) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.int64]]]:
      """Mock implementation of adapt to framework.

      Args:
          args: Test argument or fixture.
          fw: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return args

  sr.fuzzer = DummyFuzzer()  # type: ignore[assignment]
  with __import__("unittest.mock").mock.patch.object(sr, "_execute_api", return_value=1):
    res, msg = sr.verify(variants, ["x"], {"x": "Array"}, {}, shape_calc="lambda y: 1/0")
    assert res is False
    assert "Shape Calculation Error" in msg

    with __import__("unittest.mock").mock.patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      res_ok, _ = sr.verify(variants, ["x"], {"x": "Array"}, {}, shape_calc="lambda x: (1,)")
      assert res_ok is True


def test_runner_deep_compare_kind_o() -> None:
  """Verifies the behavior of runner deep compare kind o."""
  import numpy as np

  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()
  assert sr._deep_compare(np.array([object()]), np.array([object()])) is False


def test_runner_deep_compare_kind_o_match() -> None:
  """Verifies the behavior of runner deep compare kind o match."""
  import numpy as np

  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()
  assert sr._deep_compare(np.array(["a", "b"]), np.array(["a", "b"])) is True


def test_runner_deep_compare_fallback() -> None:
  """Verifies the behavior of runner deep compare fallback."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()
  assert sr._deep_compare("string", "string") is True


def test_evaluate_shape_calc_ast_branches() -> None:
  """Verify `_evaluate_shape_calc` supports safe AST nodes and rejects unsafe expressions."""
  import numpy as np
  import pytest

  from ml_switcheroo.testing.runner import _evaluate_shape_calc

  arr = np.zeros((4, 8))
  assert _evaluate_shape_calc("lambda x: x.shape", arr) == (4, 8)
  assert _evaluate_shape_calc("lambda x: [x.shape[0] + 1, x.shape[1] - 2]", arr) == [5, 6]
  assert _evaluate_shape_calc("lambda x: (x.shape[0] * 2, x.shape[1] // 2, x.shape[1] / 2, x.shape[1] % 3)", arr) == (
    8,
    4,
    4.0,
    2,
  )
  assert _evaluate_shape_calc("lambda x: tuple(x.shape[:1])", arr) == (4,)
  assert _evaluate_shape_calc("lambda x: (-1, len(x.shape))", arr) == (-1, 2)

  with pytest.raises(ValueError, match="Expected single-argument lambda"):
    _evaluate_shape_calc("x.shape", arr)
  with pytest.raises(ValueError, match="Unknown identifier"):
    _evaluate_shape_calc("lambda x: unknown_var", arr)
  with pytest.raises(ValueError, match="Private attribute"):
    _evaluate_shape_calc("lambda x: x.__class__", arr)
  with pytest.raises(ValueError, match="Unsupported AST node"):
    _evaluate_shape_calc("lambda x: {1: 2}", arr)
  with pytest.raises(ValueError, match="Unsupported AST node"):
    _evaluate_shape_calc("lambda x: x.shape[0] ** 2", arr)


def test_runner_import_error_ignored_and_runtime_error_recorded() -> None:
  """Verify `EquivalenceRunner.verify` ignores missing optional framework imports but records runtime errors."""
  from unittest.mock import patch

  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()
  variants = {"missing_fw": {"api": "missing_fw_pkg_xyz.op"}}
  ok, _ = sr.verify(variants, [])
  assert ok is True

  with patch.object(sr, "_execute_api", side_effect=TypeError("bad signature")):
    ok_err, msg_err = sr.verify({"torch": {"api": "torch.add"}}, [])
    assert ok_err is False
    assert "Crash in torch: bad signature" in msg_err
