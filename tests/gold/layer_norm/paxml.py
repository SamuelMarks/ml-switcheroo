"""Test suite for the Paxml module."""

from praxis import base_layer
from praxis.layers import normalizations
import jax.numpy as jnp


class LayerNormModel(base_layer.BaseLayer):
  """Test suite for the Layer Norm Model component."""

  normalized_shape: int = 0

  def setup(self):
    """Helper to setup."""
    self.create_child("ln", normalizations.LayerNorm.HParams(dim=self.normalized_shape))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.ln(x)
