"""Test suite for the Paxml module."""

from praxis import base_layer
from praxis.layers import normalizations
import jax.numpy as jnp


class BNModel(base_layer.BaseLayer):
  """Test suite for the B N Model component."""

  num_features: int = 0

  def setup(self):
    """Helper to setup."""
    self.create_child("bn", normalizations.BatchNorm.HParams(dim=self.num_features))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.bn(x)
