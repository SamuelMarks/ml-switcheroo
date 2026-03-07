"""Module docstring."""

import jax.numpy as jnp
from praxis.layers import activations


def relu_activation(x: jnp.ndarray) -> jnp.ndarray:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # PaxML activations are typically classes
  act = activations.ReLU.HParams().instantiate()
  return act(x)
  # </SWITCHEROO_FAILED_TO_TRANS>
