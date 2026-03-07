"""Module docstring."""

from praxis import base_layer
from praxis.layers import normalizations
import jax.numpy as jnp


class BNModel(base_layer.BaseLayer):
  """Class docstring."""

  num_features: int = 0

  def setup(self):
    """Function docstring."""
    self.create_child("bn", normalizations.BatchNorm.HParams(dim=self.num_features))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.bn(x)
