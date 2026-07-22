"""Test suite for the Fuzzer Core module."""


def test_fuzzer_core_coverage():
  """Verifies the behavior of fuzzer core coverage."""
  from ml_switcheroo.testing.fuzzer.core import InputFuzzer

  ig = InputFuzzer()
  ig.build_strategies(["shape", "axis", "mask", "indices", "alpha", "inputs"])

  class FailingAdapter:
    """Test suite for the Failing Adapter component."""

    def convert(self, x):
      """Converts ."""
      raise ValueError("fail")

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.fuzzer.core.get_adapter", return_value=FailingAdapter()
  ):
    res = ig.adapt_to_framework({"a": 1}, "jax")
    assert res["a"] == 1
  with __import__("unittest.mock").mock.patch("ml_switcheroo.testing.fuzzer.core.get_adapter", return_value=None):
    res = ig.adapt_to_framework({"a": 1}, "jax")
    assert res["a"] == 1
