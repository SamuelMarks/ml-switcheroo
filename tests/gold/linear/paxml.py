"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp

from praxis.layers import linears


class Model(base_layer.BaseLayer):
  """Test helper class Model."""

  in_features = 0
  out_features = 0

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child("linear", linears.Linear.HParams(input_dims=self.in_features, output_dims=self.out_features))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.linear(x)
