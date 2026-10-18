"""Test suite for the Flax Nnx module."""

import typing

import jax


def conditional_op(pred: typing.Any, x: typing.Any) -> typing.Any:
  """Helper to conditional op.

  Args:
      pred: Test argument or fixture.
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return jax.lax.cond(pred, lambda operand: operand * 2, lambda operand: operand + 2, x)
