"""Module docstring."""

from praxis import base_layer
from praxis.layers import convolutions
from praxis.layers import normalizations
from praxis.layers import activations
import jax.numpy as jnp


class ResidualBlock(base_layer.BaseLayer):
  """Class docstring."""

  channels: int = 0

  def setup(self):
    """Function docstring."""
    self.create_child(
      "conv1", convolutions.Conv2D.HParams(filter_shape=(3, 3, self.channels, self.channels), padding="SAME")
    )
    self.create_child("bn1", normalizations.BatchNorm.HParams(dim=self.channels))
    self.create_child("relu1", activations.ReLU.HParams())
    self.create_child(
      "conv2", convolutions.Conv2D.HParams(filter_shape=(3, 3, self.channels, self.channels), padding="SAME")
    )
    self.create_child("bn2", normalizations.BatchNorm.HParams(dim=self.channels))
    self.create_child("relu2", activations.ReLU.HParams())

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + residual
    out = self.relu2(out)
    return out
