"""Module docstring."""

from flax import nnx
import jax.numpy as jnp


class BNModel(nnx.Module):
  """Class docstring."""

  def __init__(self, num_features: int, rngs: nnx.Rngs):
    """Function docstring."""
    self.bn = nnx.BatchNorm(num_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray, use_running_average: bool = False) -> jnp.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # NNX explicitly separates training state via the use_running_average flag
    return self.bn(x, use_running_average=use_running_average)
    # </SWITCHEROO_FAILED_TO_TRANS>
