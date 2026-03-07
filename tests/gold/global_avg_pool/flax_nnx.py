"""Module docstring."""

from flax import nnx
import jax.numpy as jnp


class GAPModel(nnx.Module):
  """Class docstring."""

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Flax/JAX assumes NHWC conventionally, GAP means reducing spatial dimensions (1, 2)
    return jnp.mean(x, axis=(1, 2))
    # </SWITCHEROO_FAILED_TO_TRANS>
