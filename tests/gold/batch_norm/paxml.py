"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis import base_layer
from praxis.layers import normalizations


class BNModel(base_layer.BaseLayer):
  """Docstring."""

  num_features: int = 0

  def setup(self):
    """Helper to setup."""
    self.create_child("bn", normalizations.BatchNorm.HParams(dim=self.num_features))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.bn(x)
