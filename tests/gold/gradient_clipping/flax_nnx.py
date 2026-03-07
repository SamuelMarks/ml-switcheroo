"""Module docstring."""

import optax
from flax import nnx


def get_clipped_optimizer(learning_rate: float, max_norm: float = 1.0):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # JAX uses Optax chains
  return optax.chain(optax.clip_by_global_norm(max_norm), optax.adam(learning_rate))
  # </SWITCHEROO_FAILED_TO_TRANS>
