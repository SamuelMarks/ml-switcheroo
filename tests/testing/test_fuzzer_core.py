def test_fuzzer_core_coverage():
  from ml_switcheroo.testing.fuzzer.core import InputFuzzer

  ig = InputFuzzer()

  strat = ig.build_strategies(["shape", "axis", "mask", "indices", "alpha", "inputs"])

  # 103-104: conversion failure
  class FailingAdapter:
    def convert(self, x):
      raise ValueError("fail")

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.fuzzer.core.get_adapter", return_value=FailingAdapter()
  ):
    res = ig.adapt_to_framework({"a": 1}, "jax")
    assert res["a"] == 1

  with __import__("unittest.mock").mock.patch("ml_switcheroo.testing.fuzzer.core.get_adapter", return_value=None):
    res = ig.adapt_to_framework({"a": 1}, "jax")
    assert res["a"] == 1
