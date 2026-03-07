"""Module docstring."""

from praxis import base_layer
from praxis.layers import linears
import jax.numpy as jnp


class Model(base_layer.BaseLayer):
  """Class docstring."""

  in_features: int = 0
  out_features: int = 0

  def setup(self):
    """Function docstring."""
    self.create_child("linear", linears.Linear.HParams(input_dims=self.in_features, output_dims=self.out_features))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.linear(x)
