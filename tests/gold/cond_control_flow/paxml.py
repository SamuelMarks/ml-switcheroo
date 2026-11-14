"""PaxML / JAX golden reference for data-dependent conditional control flow."""

import jax
import jax.numpy as jnp


def conditional_op(pred: jax.Array, x: jax.Array) -> jax.Array:
  """Select ``x * 2`` or ``x + 2`` functionally via ``jax.lax.cond``.

  Args:
      pred: Scalar predicate array.
      x: Input JAX array operand.

  Returns:
      Result JAX array produced by the active branch.
  """
  res: jax.Array = jax.lax.cond(
    jnp.asarray(pred, dtype=jnp.bool_),
    lambda operand: operand * 2,
    lambda operand: operand + 2,
    x,
  )
  return res
