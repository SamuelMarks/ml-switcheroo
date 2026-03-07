"""Module docstring."""

from praxis import base_layer
from praxis.layers import stochastic
import jax.numpy as jnp


class DropoutModel(base_layer.BaseLayer):
  """Class docstring."""

  p: float = 0.5

  def setup(self):
    """Function docstring."""
    self.create_child("dropout", stochastic.Dropout.HParams(keep_prob=1.0 - self.p))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # PaxML relies on a do_eval state managed by the context
    return self.dropout(x)
    # </SWITCHEROO_FAILED_TO_TRANS>
