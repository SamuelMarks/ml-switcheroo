"""Module docstring."""

from praxis import base_layer
import jax.numpy as jnp


class FlattenModel(base_layer.BaseLayer):
  """Class docstring."""

  start_dim: int = 1

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # functional reshape
    batch_shape = x.shape[: self.start_dim]
    return x.reshape((*batch_shape, -1))
    # </SWITCHEROO_FAILED_TO_TRANS>
