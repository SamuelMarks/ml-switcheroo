"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis import base_layer
from praxis.layers import activations, linears


class MLP(base_layer.BaseLayer):
  """Stub `base_layer.BaseLayer` subclass implementing `setup` for PaxML tests."""

  in_features = 0
  hidden_features = 0
  out_features = 0

  def setup(self) -> None:
    """Execute `create_child` to support setup in PaxML tests."""
    self.create_child("fc1", linears.Linear.HParams(input_dims=self.in_features, output_dims=self.hidden_features))
    self.create_child("relu", activations.ReLU.HParams())
    self.create_child("fc2", linears.Linear.HParams(input_dims=self.hidden_features, output_dims=self.out_features))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    x = self.fc1(x)
    x = self.relu(x)
    return self.fc2(x)
