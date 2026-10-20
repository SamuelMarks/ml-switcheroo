"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp


def conditional_op(pred: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
  """Helper to conditional op.

  Args:
      pred: Test argument or fixture.
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return jax.lax.cond(pred, lambda operand: operand * 2, lambda operand: operand + 2, x)
