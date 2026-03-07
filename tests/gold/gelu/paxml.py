"""Module docstring."""

import jax.numpy as jnp
from praxis.layers import activations


def gelu_activation(x: jnp.ndarray, approximate: bool = False) -> jnp.ndarray:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # PaxML layers abstract this into HParams
  act = activations.GELU.HParams(approximate=approximate).instantiate()
  return act(x)
  # </SWITCHEROO_FAILED_TO_TRANS>
