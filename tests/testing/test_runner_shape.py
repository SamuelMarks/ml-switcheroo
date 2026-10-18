"""Test suite for the Runner Shape module."""

import typing

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo.testing.runner import EquivalenceRunner


@pytest.fixture
def runner() -> typing.Any:
  """Execute runner helper logic.

  Returns:
      Computed test result or fixture value.
  """
  r: typing.Any = EquivalenceRunner()
  r.fuzzer = MagicMock()
  return r


def test_verify_passes_correct_shape(runner: typing.Any) -> None:
  """Verifies passes correct shape.

  Args:
      runner: Test argument or fixture.
  """
  inputs: typing.Any = {"x": np.array([1, 2])}
  runner.fuzzer.build_strategies.return_value = {}
  runner.fuzzer.generate_inputs_oneshot.return_value = inputs
  import hypothesis.strategies as st

  runner.fuzzer.build_strategies.return_value = {"x": st.just(np.array([1, 2]))}
  runner.fuzzer.adapt_to_framework.side_effect = lambda d, fw: d
  variants: typing.Any = {"mock": {"api": "mock.op"}}
  with patch.object(runner, "_execute_api", return_value=inputs["x"]):
    with patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      passed: typing.Any
      msg: typing.Any
      passed, msg = runner.verify(variants, params=["x"], shape_calc="lambda x: x.shape")
  assert passed is True


def test_verify_fails_shape_mismatch(runner: typing.Any) -> None:
  """Verifies fails shape mismatch.

  Args:
      runner: Test argument or fixture.
  """
  import hypothesis.strategies as st

  inputs: typing.Any = {"x": np.zeros((2,))}
  runner.fuzzer.build_strategies.return_value = {"x": st.just(inputs["x"])}
  runner.fuzzer.adapt_to_framework.side_effect = lambda d, fw: d
  variants: typing.Any = {"mock": {"api": "mock.op"}}
  with patch.object(runner, "_execute_api", return_value=inputs["x"]):
    with patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      passed: typing.Any
      msg: typing.Any
      passed, msg = runner.verify(variants, params=["x"], shape_calc="lambda x: (3,)")
  assert passed is False
  assert "Shape Mismatch" in msg
