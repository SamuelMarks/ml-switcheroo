"""Module docstring."""

from praxis import base_layer
from praxis.layers import normalizations
import jax.numpy as jnp


class LayerNormModel(base_layer.BaseLayer):
  """Class docstring."""

  normalized_shape: int = 0

  def setup(self):
    """Function docstring."""
    self.create_child("ln", normalizations.LayerNorm.HParams(dim=self.normalized_shape))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.ln(x)
