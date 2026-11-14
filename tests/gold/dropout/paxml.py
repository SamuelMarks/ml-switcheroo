"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis import base_layer
from praxis.layers import stochastic


class DropoutModel(base_layer.BaseLayer):
  """Stub `base_layer.BaseLayer` subclass implementing `setup` for PaxML tests."""

  p = 0.5

  def setup(self) -> None:
    """Execute `create_child` to support setup in PaxML tests."""
    self.create_child("dropout", stochastic.Dropout.HParams(keep_prob=1.0 - self.p))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.dropout(x)
