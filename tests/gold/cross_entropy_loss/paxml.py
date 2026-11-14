"""Test suite for the Paxml module."""

import jax.numpy as jnp


def compute_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
  """Computes loss.

  Args:
      logits: Input `logits` parameter value (`jnp.ndarray`).
      targets: Input `targets` parameter value (`jnp.ndarray`).

  Returns:
      Resulting `jnp.ndarray` from compute loss.
  """
  return logits
