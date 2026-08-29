"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis import base_layer


class FlattenModel(base_layer.BaseLayer):
  """Docstring."""

  start_dim: int = 1

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    batch_shape = x.shape[: self.start_dim]
    return x.reshape((*batch_shape, -1))
