"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp

from praxis.layers import activations
from praxis.layers import convolutions
from praxis.layers import normalizations


class ResidualBlock(base_layer.BaseLayer):
  """Test helper class ResidualBlock."""

  channels = 0

  def setup(self) -> None:
    """Helper to setup."""
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
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + residual
    out = self.relu2(out)
    return out
