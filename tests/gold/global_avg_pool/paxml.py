"""Module docstring."""

from praxis import base_layer
import jax.numpy as jnp


class GAPModel(base_layer.BaseLayer):
  """Class docstring."""

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    # Assuming NHWC format
    return jnp.mean(x, axis=(1, 2))
