"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis import base_layer
from praxis.layers import activations, convolutions, normalizations


class ResidualBlock(base_layer.BaseLayer):
  """Stub `base_layer.BaseLayer` subclass implementing `setup` for PaxML tests."""

  channels = 0

  def setup(self) -> None:
    """Execute `create_child` to support setup in PaxML tests."""
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
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + residual
    return self.relu2(out)
