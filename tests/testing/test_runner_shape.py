"""Test suite for the Runner Shape module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo.testing.runner import EquivalenceRunner


@pytest.fixture
def runner() -> EquivalenceRunner:
  """Create a configured `EquivalenceRunner` fixture for runner shape tests.

  Returns:
      Configured `runner` fixture value (EquivalenceRunner).
  """
  r = EquivalenceRunner()
  r.fuzzer = MagicMock()
  return r


def test_verify_passes_correct_shape(runner: EquivalenceRunner, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies passes correct shape.

  Args:
      runner: Input `runner` parameter value (`EquivalenceRunner`).
      monkeypatch: Pytest monkeypatch fixture.
  """
  inputs = {"x": np.array([1, 2])}
  monkeypatch.setattr(
    runner.fuzzer.build_strategies,
    "return_value",
    {},
    raising=False,
  )
  monkeypatch.setattr(
    runner.fuzzer,
    "generate_inputs_oneshot",
    MagicMock(return_value=inputs),
    raising=False,
  )
  import hypothesis.strategies as st

  monkeypatch.setattr(
    runner.fuzzer.build_strategies,
    "return_value",
    {"x": st.just(np.array([1, 2]))},
    raising=False,
  )
  monkeypatch.setattr(
    runner.fuzzer.adapt_to_framework,
    "side_effect",
    (lambda d, fw: d),
    raising=False,
  )
  variants = {"mock": {"api": "mock.op"}}
  with (
    patch.object(runner, "_execute_api", return_value=inputs["x"]),
    patch("ml_switcheroo.testing.runner.get_adapter", return_value=None),
  ):
    passed, _msg = runner.verify(variants, params=["x"], shape_calc="lambda x: x.shape")
  assert passed is True


def test_verify_fails_shape_mismatch(runner: EquivalenceRunner, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies fails shape mismatch.

  Args:
      runner: Input `runner` parameter value (`EquivalenceRunner`).
      monkeypatch: Pytest monkeypatch fixture.
  """
  import hypothesis.strategies as st

  inputs = {"x": np.zeros((2,))}
  monkeypatch.setattr(
    runner.fuzzer.build_strategies,
    "return_value",
    {"x": st.just(inputs["x"])},
    raising=False,
  )
  monkeypatch.setattr(
    runner.fuzzer.adapt_to_framework,
    "side_effect",
    (lambda d, fw: d),
    raising=False,
  )
  variants = {"mock": {"api": "mock.op"}}
  with (
    patch.object(runner, "_execute_api", return_value=inputs["x"]),
    patch("ml_switcheroo.testing.runner.get_adapter", return_value=None),
  ):
    passed, msg = runner.verify(variants, params=["x"], shape_calc="lambda x: (3,)")
  assert passed is False
  assert "Shape Mismatch" in msg
