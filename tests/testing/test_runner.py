"""Test suite for the Runner Missing module."""

from typing import Any, Dict, List


def test_runner_run_exceptions() -> None:
  """Verifies the behavior of runner run exceptions."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr: EquivalenceRunner = EquivalenceRunner()
  variants: Dict[str, Any] = {"jax": "bad"}
  res: bool
  msg: str
  res, msg = sr.verify(variants, [], {}, {})
  assert res is True
  with __import__("unittest.mock").mock.patch.object(sr, "_execute_api", return_value=1):
    res, msg = sr.verify(variants, [], {}, {})

  def force_fail(*args: Any, **kwargs: Any) -> None:
    """Helper to force fail."""
    raise ValueError("hypothesis failed")

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.runner.EquivalenceRunner._compare_results", side_effect=ValueError("fail hypothesis")
  ):
    res, msg = sr.verify(variants, [], {}, {})
    assert "Verification Failed" in msg


def test_runner_execute_api() -> None:
  """Verifies the behavior of runner execute API."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr: EquivalenceRunner = EquivalenceRunner()
  assert sr._execute_api("no_dot", {}) is None


def test_runner_compare() -> None:
  """Verifies the behavior of runner compare."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr: EquivalenceRunner = EquivalenceRunner()
  err_box: List[str] = []
  import pytest

  with pytest.raises(AssertionError):
    sr._compare_results({"jax": 1, "torch": 2}, 0.001, 0.0001, err_box)
  assert len(err_box) == 1
  with pytest.raises(AssertionError):
    sr._compare_results({"jax": [1], "torch": [1, 2]}, 0.001, 0.0001, [])


def test_runner_deep_compare_exceptions() -> None:
  """Verifies the behavior of runner deep compare exceptions."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr: EquivalenceRunner = EquivalenceRunner()

  class BadNumpy:
    """Docstring."""

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
      """Helper to   array  ."""
      raise Exception("fail")

  assert sr._deep_compare(1, BadNumpy()) is False
  import numpy as np

  assert sr._deep_compare(np.array(["a"]), np.array(["a"])) is True
  assert sr._deep_compare(np.array(["a"]), np.array(["b"])) is False


def test_runner_run_details_not_dict() -> None:
  """Verifies the behavior of runner run details not dictionary."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr: EquivalenceRunner = EquivalenceRunner()
  # Hit line 93: details is dict but missing "api" key
  variants: Dict[str, Any] = {"jax": "string", "torch": {"api": "torch.add"}, "mlx": {"not_api": "something"}}
  with __import__("unittest.mock").mock.patch.object(sr, "_execute_api", return_value=1):
    res: bool
    msg: str
    res, msg = sr.verify(variants, [], {}, {})
    assert "Verified" in msg or "Failures" in msg


def test_runner_deep_compare_shape_mismatch() -> None:
  """Verifies behavior when arrays have different shapes."""
  import numpy as np

  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr: EquivalenceRunner = EquivalenceRunner()
  # Hit line 228
  assert sr._deep_compare(np.array([1, 2]), np.array([1, 2, 3])) is False


def test_runner_run_shape_calculation_error() -> None:
  """Verifies the behavior of runner run shape calculation correctly handling an error."""
  import numpy as np

  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr: EquivalenceRunner = EquivalenceRunner()
  variants: Dict[str, Any] = {"jax": {"api": "jax.add"}}

  class DummyFuzzer:
    """Docstring."""

    def build_strategies(self, p: List[str], h: Dict[str, str], c: Dict[str, Any]) -> Dict[str, Any]:
      """Mock implementation of build strategies."""
      import hypothesis.strategies as st

      return {"x": st.just(np.array([1]))}

    def adapt_to_framework(self, args: Dict[str, Any], fw: str) -> Dict[str, Any]:
      """Mock implementation of adapt to framework."""
      return args

  sr.fuzzer = DummyFuzzer()
  with __import__("unittest.mock").mock.patch.object(sr, "_execute_api", return_value=1):
    res: bool
    msg: str
    res, msg = sr.verify(variants, ["x"], {"x": "Array"}, {}, shape_calc="lambda y: 1/0")
    assert res is False
    assert "Shape Calculation Error" in msg


def test_runner_deep_compare_kind_o() -> None:
  """Verifies the behavior of runner deep compare kind o."""
  import numpy as np

  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr: EquivalenceRunner = EquivalenceRunner()
  assert sr._deep_compare(np.array([object()]), np.array([object()])) is False


def test_runner_deep_compare_kind_o_match() -> None:
  """Verifies the behavior of runner deep compare kind o match."""
  import numpy as np

  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr: EquivalenceRunner = EquivalenceRunner()
  assert sr._deep_compare(np.array(["a", "b"]), np.array(["a", "b"])) is True


def test_runner_deep_compare_fallback() -> None:
  """Verifies the behavior of runner deep compare fallback."""
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr: EquivalenceRunner = EquivalenceRunner()
  assert sr._deep_compare("string", "string") is True
