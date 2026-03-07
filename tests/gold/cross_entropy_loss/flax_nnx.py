"""Module docstring."""

import jax.numpy as jnp
import optax


def compute_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # Optax expects targets to be one-hot encoded usually, or uses softmax_cross_entropy_with_integer_labels
  return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
  # </SWITCHEROO_FAILED_TO_TRANS>
