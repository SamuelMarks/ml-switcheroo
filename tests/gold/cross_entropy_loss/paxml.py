"""Test suite for the Paxml module."""

import jax.numpy as jnp


def compute_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
  """Computes loss.

  Args:
      logits: The logits (jnp.ndarray) for compute loss.
      targets: The targets (jnp.ndarray) for compute loss.

  Returns:
      Resulting `jnp.ndarray` from compute loss.
  """
  return logits
