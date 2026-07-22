"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp


class GAPModel(base_layer.BaseLayer):
  """Test suite for the G A P Model component."""

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return jnp.mean(x, axis=(1, 2))
