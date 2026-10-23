"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
  """Test helper class MLP."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int, rngs: nnx.Rngs):
    """Initializes the MLP instance.

    Args:
        in_features: Number of input linear features.
        hidden_features: The hidden features (int) for   init  .
        out_features: Number of output linear features.
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.fc1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
    self.fc2 = nnx.Linear(hidden_features, out_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    x = self.fc1(x)
    x = jax.nn.relu(x)
    x = self.fc2(x)
    return x
