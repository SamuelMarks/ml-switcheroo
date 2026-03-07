"""Module docstring."""

from flax import nnx
import jax.numpy as jnp
import jax


class MLP(nnx.Module):
  """Class docstring."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int, rngs: nnx.Rngs):
    """Function docstring."""
    self.fc1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
    self.fc2 = nnx.Linear(hidden_features, out_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    x = self.fc1(x)
    x = jax.nn.relu(x)
    x = self.fc2(x)
    return x
