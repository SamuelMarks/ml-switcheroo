"""PaxML / JAX golden reference for attention score masked fill."""

import jax.numpy as jnp


def causal_mask_fill(
  scores: jnp.ndarray,
  mask: jnp.ndarray,
  value: float = -1000000000.0,
) -> jnp.ndarray:
  """Replace positions where ``mask == 0`` with ``value`` using ``jnp.where``.

  Args:
      scores: Unmasked attention score array.
      mask: Mask array where ``0`` indicates masked positions.
      value: Constant substituted where ``mask == 0``.

  Returns:
      Masked JAX score array.
  """
  res: jnp.ndarray = jnp.where(mask == 0, value, scores)
  return res
