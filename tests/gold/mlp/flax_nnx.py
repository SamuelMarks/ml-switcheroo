"""Test suite for the Flax Nnx module."""

from flax import nnx
import jax.numpy as jnp
import jax


class MLP(nnx.Module):
  """Test suite for the M L P component."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int, rngs: nnx.Rngs):
    """Initializes the MLP instance."""
    self.fc1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
    self.fc2 = nnx.Linear(hidden_features, out_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    x = self.fc1(x)
    x = jax.nn.relu(x)
    x = self.fc2(x)
    return x
