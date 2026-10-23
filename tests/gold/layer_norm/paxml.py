"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp

from praxis.layers import normalizations


class LayerNormModel(base_layer.BaseLayer):
  """Test helper class LayerNormModel."""

  normalized_shape = 0

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child("ln", normalizations.LayerNorm.HParams(dim=self.normalized_shape))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.ln(x)
