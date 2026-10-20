"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):  # type: ignore[misc]
  """Test helper class MLP."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int, rngs: nnx.Rngs):
    """Initializes the MLP instance.

    Args:
        in_features: Test argument or fixture.
        hidden_features: Test argument or fixture.
        out_features: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    self.fc1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
    self.fc2 = nnx.Linear(hidden_features, out_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    x = self.fc1(x)
    x = jax.nn.relu(x)
    x = self.fc2(x)
    return x
