"""Module docstring."""

import jax
import jax.numpy as jnp


def conditional_op(pred: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # JAX requires jax.lax.cond for compiled dynamic control flow
  return jax.lax.cond(pred, lambda operand: operand * 2, lambda operand: operand + 2, x)
  # </SWITCHEROO_FAILED_TO_TRANS>
