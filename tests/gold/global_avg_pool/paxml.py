"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis import base_layer


class GAPModel(base_layer.BaseLayer):
  """Minimal `base_layer.BaseLayer` test double used in PaxML tests."""

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return jnp.mean(x, axis=(1, 2))
