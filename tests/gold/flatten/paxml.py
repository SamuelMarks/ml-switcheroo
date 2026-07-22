"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp


class FlattenModel(base_layer.BaseLayer):
  """Test suite for the Flatten Model component."""

  start_dim: int = 1

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    batch_shape = x.shape[: self.start_dim]
    return x.reshape((*batch_shape, -1))
