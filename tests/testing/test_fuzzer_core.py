"""Test suite for the Fuzzer Core module."""

from typing import Any, Dict


def test_fuzzer_core_coverage() -> None:
  """Verifies the behavior of fuzzer core coverage."""
  from ml_switcheroo.testing.fuzzer.core import InputFuzzer

  ig: InputFuzzer = InputFuzzer()
  ig.build_strategies(["shape", "axis", "mask", "indices", "alpha", "inputs"])

  class FailingAdapter:
    """Docstring."""

    def convert(self, x: Any) -> Any:
      """Converts ."""
      raise ValueError("fail")

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.fuzzer.core.get_adapter", return_value=FailingAdapter()
  ):
    res: Dict[str, Any] = ig.adapt_to_framework({"a": 1}, "jax")
    assert res["a"] == 1
  with __import__("unittest.mock").mock.patch("ml_switcheroo.testing.fuzzer.core.get_adapter", return_value=None):
    res2: Dict[str, Any] = ig.adapt_to_framework({"a": 1}, "jax")
    assert res2["a"] == 1
