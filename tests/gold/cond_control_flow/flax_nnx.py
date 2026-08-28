"""Test suite for the Flax Nnx module."""

import jax
import typing


def conditional_op(pred: typing.Any, x: typing.Any) -> typing.Any:
  """Helper to conditional op."""
  return jax.lax.cond(pred, lambda operand: operand * 2, lambda operand: operand + 2, x)
