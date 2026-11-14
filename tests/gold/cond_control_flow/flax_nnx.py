"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp


def conditional_op(pred: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
  """Execute `cond` to support conditional operation in flax NNX tests.

  Args:
      pred: Input `pred` parameter value (`jnp.ndarray`).
      x: Primary input tensor or array operand.

  Returns:
      Resulting `jnp.ndarray` from conditional op.
  """
  return jax.lax.cond(pred, lambda operand: operand * 2, lambda operand: operand + 2, x)
