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
  runner.fuzzer.build_strategies.return_value = {}  # type: ignore[attr-defined]
  runner.fuzzer.generate_inputs_oneshot.return_value = inputs  # type: ignore[attr-defined]
  import hypothesis.strategies as st

  runner.fuzzer.build_strategies.return_value = {"x": st.just(np.array([1, 2]))}  # type: ignore[attr-defined]
  runner.fuzzer.adapt_to_framework.side_effect = lambda d, fw: d  # type: ignore[attr-defined]
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
  runner.fuzzer.build_strategies.return_value = {"x": st.just(inputs["x"])}  # type: ignore[attr-defined]
  runner.fuzzer.adapt_to_framework.side_effect = lambda d, fw: d  # type: ignore[attr-defined]
  variants = {"mock": {"api": "mock.op"}}
  with patch.object(runner, "_execute_api", return_value=inputs["x"]):
    with patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      passed, msg = runner.verify(variants, params=["x"], shape_calc="lambda x: (3,)")
  assert passed is False
  assert "Shape Mismatch" in msg
