def test_runner_run_exceptions():
  from ml_switcheroo.testing.runner import EquivalenceRunner
  import hypothesis

  sr = EquivalenceRunner()

  variants = {"jax": "bad"}  # Not a dict (line 59)
  res, msg = sr.verify(variants, [], {}, {})
  assert res is True

  with __import__("unittest.mock").mock.patch.object(sr, "_execute_api", return_value=1):
    res, msg = sr.verify(variants, [], {}, {})

  # 111-113: run_check exception
  # Instead of mocking given, we can mock run_check internally, or we can just make it fail hypothesis execution
  def force_fail(*args, **kwargs):
    raise ValueError("hypothesis failed")

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.runner.EquivalenceRunner._compare_results", side_effect=ValueError("fail hypothesis")
  ):
    res, msg = sr.verify(variants, [], {}, {})
    assert "Verification Failed" in msg


def test_runner_execute_api():
  from ml_switcheroo.testing.runner import EquivalenceRunner
  import logging

  sr = EquivalenceRunner()
  assert sr._execute_api("no_dot", {}) is None


def test_runner_compare():
  from ml_switcheroo.testing.runner import EquivalenceRunner
  import logging

  sr = EquivalenceRunner()

  err_box = []
  import pytest

  with pytest.raises(AssertionError):
    sr._compare_results({"jax": 1, "torch": 2}, 1e-3, 1e-4, err_box)
  assert len(err_box) == 1

  with pytest.raises(AssertionError):
    sr._compare_results({"jax": [1], "torch": [1, 2]}, 1e-3, 1e-4, [])


def test_runner_deep_compare_exceptions():
  from ml_switcheroo.testing.runner import EquivalenceRunner
  import logging

  sr = EquivalenceRunner()

  class BadNumpy:
    def __array__(self, *args, **kwargs):
      raise Exception("fail")

  assert sr._deep_compare(1, BadNumpy()) is False

  import numpy as np

  assert sr._deep_compare(np.array(["a"]), np.array(["a"])) is True
  assert sr._deep_compare(np.array(["a"]), np.array(["b"])) is False


def test_runner_run_details_not_dict():
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()

  variants = {"jax": "string", "torch": {"api": "torch.add"}}

  with __import__("unittest.mock").mock.patch.object(sr, "_execute_api", return_value=1):
    res, msg = sr.verify(variants, [], {}, {})
    assert "Verified" in msg or "Failures" in msg


def test_runner_run_shape_calculation_error():
  from ml_switcheroo.testing.runner import EquivalenceRunner
  import numpy as np

  sr = EquivalenceRunner()

  variants = {"jax": {"api": "jax.add"}}

  class DummyFuzzer:
    def build_strategies(self, p, h, c):
      import hypothesis.strategies as st

      return {"x": st.just(np.array([1]))}

    def adapt_to_framework(self, args, fw):
      return args

  sr.fuzzer = DummyFuzzer()

  with __import__("unittest.mock").mock.patch.object(sr, "_execute_api", return_value=1):
    # 99-100: We need x in inputs and a shape_calc that raises exception
    res, msg = sr.verify(variants, ["x"], {"x": "Array"}, {}, shape_calc="lambda y: 1/0")
    assert res is False
    assert "Shape Calculation Error" in msg


def test_runner_deep_compare_kind_o():
  from ml_switcheroo.testing.runner import EquivalenceRunner
  import numpy as np

  sr = EquivalenceRunner()

  assert sr._deep_compare(np.array([object()]), np.array([object()])) is False


def test_runner_deep_compare_kind_o_match():
  from ml_switcheroo.testing.runner import EquivalenceRunner
  import numpy as np

  sr = EquivalenceRunner()

  assert sr._deep_compare(np.array(["a", "b"]), np.array(["a", "b"])) is True


def test_runner_deep_compare_fallback():
  from ml_switcheroo.testing.runner import EquivalenceRunner

  sr = EquivalenceRunner()
  assert sr._deep_compare("string", "string") is True
