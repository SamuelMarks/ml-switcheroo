"""Test suite for the Paxml module."""

from praxis import base_layer

import jax.numpy as jnp


class GAPModel(base_layer.BaseLayer):
  """Test helper class GAPModel."""

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return jnp.mean(x, axis=(1, 2))
