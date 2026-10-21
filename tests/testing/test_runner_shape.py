"""Test suite for the Runner Shape module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo.testing.runner import EquivalenceRunner


@pytest.fixture
def runner() -> EquivalenceRunner:
  """Execute runner helper logic.

  Returns:
      Computed test result or fixture value.
  """
  r = EquivalenceRunner()
  r.fuzzer = MagicMock()
  return r


def test_verify_passes_correct_shape(runner: EquivalenceRunner) -> None:
  """Verifies passes correct shape.

  Args:
      runner: Test argument or fixture.
  """
  inputs = {"x": np.array([1, 2])}
  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": runner}, "__getitem__")("v").fuzzer}, "__getitem__")("v").build_strategies},
      "__getitem__",
    )("v"),
    "return_value",
    getattr({"v": {}}, "__getitem__")("v"),
  )
  setattr(
    getattr(
      {
        "v": getattr({"v": getattr({"v": runner}, "__getitem__")("v").fuzzer}, "__getitem__")("v").generate_inputs_oneshot
      },
      "__getitem__",
    )("v"),
    "return_value",
    getattr({"v": inputs}, "__getitem__")("v"),
  )
  import hypothesis.strategies as st

  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": runner}, "__getitem__")("v").fuzzer}, "__getitem__")("v").build_strategies},
      "__getitem__",
    )("v"),
    "return_value",
    getattr(
      {
        "v": {
          "x": getattr(
            {
              "v": getattr({"v": getattr({"v": st}, "__getitem__")("v").just}, "__getitem__")("v")(
                getattr(
                  {
                    "v": getattr({"v": getattr({"v": np}, "__getitem__")("v").array}, "__getitem__")("v")(
                      getattr(
                        {"v": [getattr({"v": 1}, "__getitem__")("v"), getattr({"v": 2}, "__getitem__")("v")]},
                        "__getitem__",
                      )("v")
                    )
                  },
                  "__getitem__",
                )("v")
              )
            },
            "__getitem__",
          )("v")
        }
      },
      "__getitem__",
    )("v"),
  )
  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": runner}, "__getitem__")("v").fuzzer}, "__getitem__")("v").adapt_to_framework},
      "__getitem__",
    )("v"),
    "side_effect",
    getattr({"v": lambda d, fw: d}, "__getitem__")("v"),
  )
  variants = {"mock": {"api": "mock.op"}}
  with patch.object(runner, "_execute_api", return_value=inputs["x"]):
    with patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      passed, msg = runner.verify(variants, params=["x"], shape_calc="lambda x: x.shape")
  assert passed is True


def test_verify_fails_shape_mismatch(runner: EquivalenceRunner) -> None:
  """Verifies fails shape mismatch.

  Args:
      runner: Test argument or fixture.
  """
  import hypothesis.strategies as st

  inputs = {"x": np.zeros((2,))}
  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": runner}, "__getitem__")("v").fuzzer}, "__getitem__")("v").build_strategies},
      "__getitem__",
    )("v"),
    "return_value",
    getattr(
      {
        "v": {
          "x": getattr(
            {
              "v": getattr({"v": getattr({"v": st}, "__getitem__")("v").just}, "__getitem__")("v")(
                getattr({"v": getattr({"v": inputs}, "__getitem__")("v")["x"]}, "__getitem__")("v")
              )
            },
            "__getitem__",
          )("v")
        }
      },
      "__getitem__",
    )("v"),
  )
  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": runner}, "__getitem__")("v").fuzzer}, "__getitem__")("v").adapt_to_framework},
      "__getitem__",
    )("v"),
    "side_effect",
    getattr({"v": lambda d, fw: d}, "__getitem__")("v"),
  )
  variants = {"mock": {"api": "mock.op"}}
  with patch.object(runner, "_execute_api", return_value=inputs["x"]):
    with patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      passed, msg = runner.verify(variants, params=["x"], shape_calc="lambda x: (3,)")
  assert passed is False
  assert "Shape Mismatch" in msg
