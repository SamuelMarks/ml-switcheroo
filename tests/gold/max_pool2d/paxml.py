"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis import base_layer
from praxis.layers import poolings


class MaxPoolModel(base_layer.BaseLayer):
  """Stub `base_layer.BaseLayer` subclass implementing `setup` for PaxML tests."""

  kernel_size = 2
  stride = 2

  def setup(self) -> None:
    """Execute `create_child` to support setup in PaxML tests."""
    self.create_child(
      "pool",
      poolings.Pooling.HParams(
        window_shape=(self.kernel_size, self.kernel_size), window_stride=(self.stride, self.stride), pooling_type="MAX"
      ),
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.pool(x)
