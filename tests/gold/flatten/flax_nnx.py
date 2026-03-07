"""Module docstring."""

from flax import nnx
import jax.numpy as jnp


class FlattenModel(nnx.Module):
  """Class docstring."""

  def __init__(self, start_dim: int = 1):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # NNX/JAX often relies on dynamic tensor reshaping rather than stateful layers
    self.start_dim = start_dim
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    # Flatten all dimensions after start_dim
    batch_shape = x.shape[: self.start_dim]
    return x.reshape((*batch_shape, -1))
