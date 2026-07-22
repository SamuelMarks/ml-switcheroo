"""Test suite for the Paxml module."""

from praxis import base_layer
from praxis.layers import stochastic
import jax.numpy as jnp


class DropoutModel(base_layer.BaseLayer):
  """Test suite for the Dropout Model component."""

  p: float = 0.5

  def setup(self):
    """Helper to setup."""
    self.create_child("dropout", stochastic.Dropout.HParams(keep_prob=1.0 - self.p))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.dropout(x)
