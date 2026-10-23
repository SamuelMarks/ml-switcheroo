"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp

from praxis.layers import normalizations


class BNModel(base_layer.BaseLayer):
  """Test helper class BNModel."""

  num_features = 0

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child("bn", normalizations.BatchNorm.HParams(dim=self.num_features))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.bn(x)
