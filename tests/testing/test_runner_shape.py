"""
Tests for EquivalenceRunner Shape verification logic.

Verifies:
1. Valid shape calculation allows passing.
2. Invalid shape calculation causes failure.
3. Bad lambda syntax causes failure.
4. Argument unpacking order into lambda.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from ml_switcheroo.testing.runner import EquivalenceRunner


@pytest.fixture
def runner():
  r = EquivalenceRunner()
  # Mock fuzzer to return deterministic arrays
  r.fuzzer = MagicMock()
  return r


def test_verify_passes_correct_shape(runner):
  """
  Scenario: Op returns shape (2,). Lambda expects (2,).
  """
  # Setup Inputs
  inputs = {"x": np.array([1, 2])}
  runner.fuzzer.generate_inputs.return_value = inputs
  runner.fuzzer.adapt_to_framework.return_value = inputs

  # Setup Variants (Mock execution returns keys in inputs directly)
  variants = {"mock": {"api": "mock.op"}}

  # Mock Execution
  with patch.object(runner, "_execute_api", return_value=inputs["x"]):
    # Mock Normalization (Identity)
    with patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      # Execute with shape calc
      passed, msg = runner.verify(variants, params=["x"], shape_calc="lambda x: x.shape")

      # Must pass because x.shape is exactly what we returned
  assert passed is True
  assert "Output Matched" not in msg  # Actually skips compare if <2 results, wait..
  # If len(results) is 1, it returns "Skipped".
  # But shape check happens BEFORE "Skipped".
  # If shape check passes, loop continues to comparison logic.
  # If 1 variant, returns (True, "Skipped").
  # If shape check failed, it would return (False, "Shape Mismatch").

  assert "Skipped" in msg


def test_verify_fails_shape_mismatch(runner):
  """
  Scenario: Op returns shape (2,). Lambda expects (3,).
  """
  inputs = {"x": np.zeros((2,))}
  runner.fuzzer.generate_inputs.return_value = inputs
  runner.fuzzer.adapt_to_framework.return_value = inputs
  variants = {"mock": {"api": "mock.op"}}

  with patch.object(runner, "_execute_api", return_value=inputs["x"]):
    with patch("ml_switcheroo.testing.runner.get_adapter", return_value=None):
      # Lambda expects shape (3,) literal
      passed, msg = runner.verify(
        variants,
        params=["x"],
        # Lambda returns tuple (3,)
        shape_calc="lambda x: (3,)",
      )

  assert passed is False
  assert "Shape Mismatch" in msg
  assert "(2,) != (3,)" in msg


def test_verify_fails_bad_lambda_syntax(runner):
  """
  Scenario: SyntaxError in lambda string.
  """
  variants = {"mock": {"api": "foo"}}
  runner.fuzzer.generate_inputs.return_value = {"x": 1}

  # We need execute to succeed to reach shape check
  with patch.object(runner, "_execute_api", return_value=np.array([1])):
    passed, msg = runner.verify(variants, ["x"], shape_calc="lambda x: x broken syntax")

  assert passed is False
  assert "Shape Calculation Error" in msg


def test_verify_multiple_arguments_order(runner):
  """
  Scenario: Params ['a', 'b']. Lambda `lambda a, b: ...`
  Verify arguments are passed in correct positional order.
  """
  inputs = {"a": np.zeros((1,)), "b": np.zeros((5,))}
  # Fuzzer returns unordered dict logic usually? Or dependent on implementation.
  # generate_inputs returns dict.
  runner.fuzzer.generate_inputs.return_value = inputs
  runner.fuzzer.adapt_to_framework.return_value = inputs

  variants = {"mock": {"api": "f"}}

  # Return 'a'
  with patch.object(runner, "_execute_api", return_value=inputs["a"]):
    # Shape calc verifies that arg[0] is 'a' (shape (1,))
    # If order was reversed, arg[0] would be 'b' (shape (5,))

    passed, msg = runner.verify(
      variants,
      params=["a", "b"],
      # Expect first arg to have shape (1,)
      shape_calc="lambda x, y: x.shape",
    )

  assert passed is True
