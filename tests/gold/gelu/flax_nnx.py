"""Module docstring."""

import jax.numpy as jnp
import jax


def gelu_activation(x: jnp.ndarray, approximate: bool = False) -> jnp.ndarray:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # JAX uses boolean approximate
  return jax.nn.gelu(x, approximate=approximate)
  # </SWITCHEROO_FAILED_TO_TRANS>
