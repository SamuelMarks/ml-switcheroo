"""Module docstring."""

from praxis import base_layer
from praxis.layers import linears
from praxis.layers import activations
import jax.numpy as jnp


class MLP(base_layer.BaseLayer):
  """Class docstring."""

  in_features: int = 0
  hidden_features: int = 0
  out_features: int = 0

  def setup(self):
    """Function docstring."""
    self.create_child("fc1", linears.Linear.HParams(input_dims=self.in_features, output_dims=self.hidden_features))
    self.create_child("relu", activations.ReLU.HParams())
    self.create_child("fc2", linears.Linear.HParams(input_dims=self.hidden_features, output_dims=self.out_features))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x
