"""Test suite for the Runner Shape module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from ml_switcheroo.testing.runner import EquivalenceRunner


@pytest.fixture
def runner():
  """Provides a mock runner for testing."""
  r = EquivalenceRunner()
  r.fuzzer = MagicMock()
  return r


def test_verify_passes_correct_shape(runner):
  """Verifies passes correct shape."""
  inputs = {"x": np.array([1, 2])}
  runner.fuzzer.build_strategies.return_value = {}
  runner.fuzzer.generate_inputs_oneshot.return_value = inputs
  import hypothesis.strategies as st

  runner.fuzzer.build_strategies.return_value = {"x": st.just(np.array([1, 2]))}
  runner.fuzzer.adapt_to_framework.side_effect = lambda d, fw: d
  variants = {"mock": {"api": "mock.op"}}
  with patch.object(runner, "_execute_api", return_value=inputs["x"]):
    with patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      (passed, msg) = runner.verify(variants, params=["x"], shape_calc="lambda x: x.shape")
  assert passed is True


def test_verify_fails_shape_mismatch(runner):
  """Verifies fails shape mismatch."""
  import hypothesis.strategies as st

  inputs = {"x": np.zeros((2,))}
  runner.fuzzer.build_strategies.return_value = {"x": st.just(inputs["x"])}
  runner.fuzzer.adapt_to_framework.side_effect = lambda d, fw: d
  variants = {"mock": {"api": "mock.op"}}
  with patch.object(runner, "_execute_api", return_value=inputs["x"]):
    with patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      (passed, msg) = runner.verify(variants, params=["x"], shape_calc="lambda x: (3,)")
  assert passed is False
  assert "Shape Mismatch" in msg
