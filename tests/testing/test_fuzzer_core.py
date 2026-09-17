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


def test_fuzzer_core_inferred_type_not_float() -> None:
  """Verifies the behavior when inferred type is not float."""
  from unittest.mock import patch

  from ml_switcheroo.testing.fuzzer.core import InputFuzzer

  ig: InputFuzzer = InputFuzzer()
  # Explicit hint to cover branch 51->81
  strategies_explicit: Dict[str, Any] = ig.build_strategies(["x"], hints={"x": "float"})
  assert "x" in strategies_explicit

  with patch("ml_switcheroo.testing.fuzzer.core.guess_dtype_by_name", return_value="custom_type"):
    strategies: Dict[str, Any] = ig.build_strategies(["unknown_param"])
    assert "unknown_param" in strategies
