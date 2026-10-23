"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class Model(nnx.Module):
  """Test helper class Model."""

  def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
    """Initializes the Model instance.

    Args:
        in_features: Number of input linear features.
        out_features: Number of output linear features.
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.linear(x)
