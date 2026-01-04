"""
Tests for EquivalenceRunner Shape verification logic.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from ml_switcheroo.testing.runner import EquivalenceRunner


@pytest.fixture
def runner():
  r = EquivalenceRunner()
  # Mock fuzzer
  r.fuzzer = MagicMock()
  return r


def test_verify_passes_correct_shape(runner):
  """
  Scenario: Op returns shape (2,). Lambda expects (2,).
  """
  # Setup Inputs
  inputs = {"x": np.array([1, 2])}
  runner.fuzzer.build_strategies.return_value = {}
  runner.fuzzer.generate_inputs_oneshot.return_value = inputs  # Just in case
  # Note: Hypothesis loop in verify calls build_strategies. Match mocked behavior.

  # Since verify logic is inside hypothesis @given loop, mocking hypothesis behavior completely
  # is hard in unit test. We can just test _execute_api directly or integration test?
  # Actually, the runner logic handles exception inside loop.

  # For unit test of `shape_calc` logic, we can verify that manual patching of _execute_api
  # works if we target the *internal logic*.
  # However, runner.verify() uses @given so mocking `_execute_api` only works
  # if hypothesis actually runs the decorated function body.
  # With @settings(max_examples=1), it should run once.

  # But runner.fuzzer needs to return valid strategies.
  import hypothesis.strategies as st

  runner.fuzzer.build_strategies.return_value = {"x": st.just(np.array([1, 2]))}
  runner.fuzzer.adapt_to_framework.side_effect = lambda d, fw: d

  variants = {"mock": {"api": "mock.op"}}

  # Mock Execution
  # Patch _execute_api (using correct name now)
  with patch.object(runner, "_execute_api", return_value=inputs["x"]):
    with patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      passed, msg = runner.verify(
        variants,
        params=["x"],
        shape_calc="lambda x: x.shape",
      )

  assert passed is True
  # Skipped because <2 variants (only 'mock') BUT shape check runs inside loop
  # The runner returns True if no assertion error


def test_verify_fails_shape_mismatch(runner):
  """
  Scenario: Op returns shape (2,). Lambda expects (3,).
  """
  import hypothesis.strategies as st

  inputs = {"x": np.zeros((2,))}
  runner.fuzzer.build_strategies.return_value = {"x": st.just(inputs["x"])}
  runner.fuzzer.adapt_to_framework.side_effect = lambda d, fw: d
  variants = {"mock": {"api": "mock.op"}}

  with patch.object(runner, "_execute_api", return_value=inputs["x"]):
    with patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      # Lambda expects shape (3,)
      passed, msg = runner.verify(
        variants,
        params=["x"],
        shape_calc="lambda x: (3,)",
      )

  assert passed is False
  assert "Shape Mismatch" in msg
