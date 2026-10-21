"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp

from praxis.layers import stochastic


class DropoutModel(base_layer.BaseLayer):
  """Test helper class DropoutModel."""

  p = 0.5

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child("dropout", stochastic.Dropout.HParams(keep_prob=1.0 - self.p))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.dropout(x)
