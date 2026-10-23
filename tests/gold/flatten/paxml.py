"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp


class FlattenModel(base_layer.BaseLayer):
  """Test helper class FlattenModel."""

  start_dim = 1

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    batch_shape = x.shape[: self.start_dim]
    return x.reshape((*batch_shape, -1))
