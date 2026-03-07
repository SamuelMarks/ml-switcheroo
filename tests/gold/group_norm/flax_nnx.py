"""Module docstring."""

from flax import nnx
import jax.numpy as jnp


class GNModel(nnx.Module):
  """Class docstring."""

  def __init__(self, num_groups: int, num_channels: int, rngs: nnx.Rngs):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Flax/JAX GroupNorm doesn't strictly require num_channels ahead of time usually, just num_groups
    self.gn = nnx.GroupNorm(num_groups=num_groups, rngs=rngs)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.gn(x)
