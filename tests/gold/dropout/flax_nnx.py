"""Module docstring."""

from flax import nnx
import jax.numpy as jnp


class DropoutModel(nnx.Module):
  """Class docstring."""

  def __init__(self, p: float = 0.5, rngs: nnx.Rngs = None):
    """Function docstring."""
    self.dropout = nnx.Dropout(p, rngs=rngs)

  def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # NNX requires threading a deterministic flag to bypass dropout during eval
    return self.dropout(x, deterministic=deterministic)
    # </SWITCHEROO_FAILED_TO_TRANS>
